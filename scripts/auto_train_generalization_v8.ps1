[CmdletBinding()]
param(
    [int]$TotalGames = 100,
    [int]$StartAtGames = 0,
    # A multiple of four gives every map size equal opportunity per stage.
    [int]$GamesPerStage = 16,
    [int[]]$Seeds = @(),
    [int]$RandomSeedCount = 4,
    # Keep per-stage promotion bounded. Run 24x24 as a separate periodic audit;
    # the training environment still samples 12/16/24 every stage.
    [int[]]$EvaluationMapSizes = @(12, 16),
    [int]$ReplayTimeoutSeconds = 300,
    [string]$ConfigName = "conv_teacher_bc_dagger_v8_generalization",
    [string]$LeagueName = "auto_league_dagger_v8_generalization",
    [string]$InitialAgent = "",
    [switch]$AccumulateShadowLearner,
    [int]$EvaluationEveryGames = 16,
    [string[]]$TrainOverrides = @(),
    [switch]$TrainOnly,
    [switch]$EvalOnly,
    [switch]$ContinueOnReplayFailure
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Normalize-ProcessPath {
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $CurrentPath) {
        $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $CurrentPath, "Process")
}

Normalize-ProcessPath

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LuxCli = Join-Path $ProjectRoot "node_modules\.bin\lux-ai-2021.CMD"
$ReplayConverter = Join-Path $PSScriptRoot "convert_replay_stateful.js"
$NodeBin = "C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
$LeagueRoot = Join-Path $ProjectRoot ("outputs\{0}" -f $LeagueName)
$BestAgent = Join-Path $LeagueRoot "best_agent"
$LearnerAgent = Join-Path $LeagueRoot "learner_agent"

# This path is a fixed benchmark and is never a copy destination.
$Stage400Agent = Join-Path $ProjectRoot "outputs\auto_league_dagger_v7_16x16\best_agent"
$InitialAgentPath = if ($InitialAgent) {
    if ([IO.Path]::IsPathRooted($InitialAgent)) {
        $InitialAgent
    } else {
        Join-Path $ProjectRoot $InitialAgent
    }
} else {
    $Stage400Agent
}
$V4Agent = Join-Path $ProjectRoot "outputs\auto_league_dagger_v4_16x16\learner_agent"
$FirstAgent = Join-Path $ProjectRoot "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research"
$Opponents = @(
    @("stage400", $Stage400Agent),
    @("v4_stage350", $V4Agent),
    @("first", $FirstAgent)
)

function Assert-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label was not found: $Path" }
}

function New-RandomSeeds([int]$Count) {
    $Generated = [System.Collections.Generic.HashSet[int]]::new()
    while ($Generated.Count -lt $Count) {
        $Seed = Get-Random -Minimum 1 -Maximum 2147483647
        [void]$Generated.Add($Seed)
    }
    return @($Generated)
}

function Resolve-EvaluationSeeds([int[]]$ExplicitSeeds, [int]$RandomCount) {
    if ($RandomCount -gt 0) {
        return New-RandomSeeds $RandomCount
    }
    if (-not $ExplicitSeeds -or $ExplicitSeeds.Count -eq 0) {
        throw "No evaluation seeds specified. Use -RandomSeedCount N or pass -Seeds with -RandomSeedCount 0."
    }
    return $ExplicitSeeds
}

function Copy-Agent(
    [string]$Source,
    [string]$Destination,
    [string]$Weights,
    [string]$ResolvedModelConfig = ""
) {
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
    $RuntimeFiles = @(
        "lux_ai\nns\__init__.py",
        "lux_ai\nns\models.py",
        "lux_ai\rl_agent\rl_agent.py",
        "lux_ai\rl_agent\gate_policy.py",
        "lux_ai\rl_agent\learned_intervention_gate.py"
    )
    foreach ($RelativePath in $RuntimeFiles) {
        $RuntimeSource = Join-Path $ProjectRoot $RelativePath
        if (Test-Path -LiteralPath $RuntimeSource) {
            $RuntimeDestination = Join-Path $Destination $RelativePath
            $RuntimeDestinationDir = Split-Path -Parent $RuntimeDestination
            New-Item -ItemType Directory -Path $RuntimeDestinationDir -Force | Out-Null
            Copy-Item -LiteralPath $RuntimeSource -Destination $RuntimeDestination -Force
        }
    }
    $RlDir = Join-Path $Destination "lux_ai\rl_agent"
    Get-ChildItem -LiteralPath $RlDir -Filter "*.pt" -File | Remove-Item -Force
    Copy-Item -LiteralPath $Weights -Destination (Join-Path $RlDir "candidate_weights.pt") -Force
    $ConfigPath = Join-Path $RlDir "config.yaml"
    if ($ResolvedModelConfig -and (Test-Path -LiteralPath $ResolvedModelConfig)) {
        Copy-Item -LiteralPath $ResolvedModelConfig -Destination $ConfigPath -Force
    }
    $ModelConfig = (Get-Content -LiteralPath $ConfigPath -Raw) `
        -replace '(?m)^checkpoint_file:\s*\S+', 'checkpoint_file: candidate_weights.pt'
    [IO.File]::WriteAllText(
        $ConfigPath,
        $ModelConfig,
        [Text.UTF8Encoding]::new($false)
    )
    $AgentConfigPath = Join-Path $RlDir "rl_agent_config.yaml"
    $AgentConfig = Get-Content -LiteralPath $AgentConfigPath -Raw
    $AgentConfig = $AgentConfig -replace (
        '(?ms)^data_augmentations:\s*\r?\n(?:[ \t]+.*(?:\r?\n|$))*'
    ), "data_augmentations: []`r`n"
    [IO.File]::WriteAllText(
        $AgentConfigPath,
        $AgentConfig,
        [Text.UTF8Encoding]::new($false)
    )
}

function Test-Replay([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        $Replay = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $null -ne $Replay -and (Get-Item -LiteralPath $Path).Length -gt 0
    } catch { return $false }
}

function Invoke-Game(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [int]$MapSize,
    [string]$Output
) {
    if (Test-Replay $Output) {
        Write-Host "Replay complete; skipping $([IO.Path]::GetFileName($Output))"
        return
    }
    if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output -Force }
    $CommandReplay = "$Output.commands.json"
    if (Test-Path -LiteralPath $CommandReplay) { Remove-Item -LiteralPath $CommandReplay -Force }
    $Arguments = @(
        "`"$Player0`"", "`"$Player1`"", "--python", "`"$Python`"",
        "--seed", $Seed, "--loglevel", "0", "--memory", "8000",
        "--maxtime", "60000", "--width", $MapSize, "--height", $MapSize,
        "--storeLogs=true", "--statefulReplay=false", "--out", "`"$CommandReplay`""
    )
    $StdoutLog = "$CommandReplay.stdout.log"
    $StderrLog = "$CommandReplay.stderr.log"
    Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue
    $Process = Start-Process -FilePath $LuxCli -ArgumentList $Arguments -PassThru `
        -NoNewWindow -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    $Deadline = [DateTime]::UtcNow.AddSeconds($ReplayTimeoutSeconds)
    $ReplayReady = $false
    while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
        Start-Sleep -Seconds 2
        $OutputText = if (Test-Path -LiteralPath $StdoutLog) {
            Get-Content -LiteralPath $StdoutLog -Raw
        } else { "" }
        if ($OutputText -match 'timed out after|agent [01].*exception') {
            & taskkill.exe /PID $Process.Id /T /F | Out-Null
            $Tail = ($OutputText -split "`r?`n" | Select-Object -Last 12) -join "`n"
            throw "Agent failed during replay: $Output`n$Tail"
        }
        if (Test-Replay $CommandReplay) {
            $ReplayReady = $true
            if (-not $Process.HasExited) {
                try { & taskkill.exe /PID $Process.Id /T /F | Out-Null } catch {}
            }
            break
        }
    }
    if (-not $ReplayReady -and -not $Process.HasExited -and (Test-Replay $CommandReplay)) {
        $ReplayReady = $true
        try { & taskkill.exe /PID $Process.Id /T /F | Out-Null } catch {}
    }
    if (-not $ReplayReady -and -not $Process.HasExited) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
        if (Test-Path -LiteralPath $CommandReplay) { Remove-Item -LiteralPath $CommandReplay -Force }
        throw "Replay timed out after $ReplayTimeoutSeconds seconds: $Output"
    }
    if (-not (Test-Replay $CommandReplay)) { throw "Command replay failed: $CommandReplay" }

    Write-Host "Converting replay to stateful JSON..."
    $ConvertProcess = Start-Process -FilePath "node" -ArgumentList @(
        "`"$ReplayConverter`"", "`"$CommandReplay`"", "`"$Output`""
    ) -PassThru -NoNewWindow -RedirectStandardOutput "$Output.convert.stdout.log" `
        -RedirectStandardError "$Output.convert.stderr.log"
    if (-not $ConvertProcess.WaitForExit(300000)) {
        & taskkill.exe /PID $ConvertProcess.Id /T /F | Out-Null
        throw "Stateful replay conversion timed out: $Output"
    }
    if (-not (Test-Replay $Output)) {
        $ConvertError = Get-Content "$Output.convert.stderr.log" -Raw -ErrorAction SilentlyContinue
        throw "Stateful replay conversion failed: $Output`n$ConvertError"
    }
}

Assert-Path $Python "Virtual-environment Python"
Assert-Path $LuxCli "Lux CLI"
Assert-Path $ReplayConverter "Stateful replay converter"
foreach ($Opponent in $Opponents) {
    Assert-Path (Join-Path $Opponent[1] "main.py") ("Opponent {0}" -f $Opponent[0])
}
$Stage400Weights = Join-Path $Stage400Agent "lux_ai\rl_agent\candidate_weights.pt"
Assert-Path $Stage400Weights "Stage 400 weights"
$Stage400Hash = (Get-FileHash -LiteralPath $Stage400Weights -Algorithm SHA256).Hash
Assert-Path (Join-Path $InitialAgentPath "main.py") "Initial agent"
Assert-Path (Join-Path $InitialAgentPath "lux_ai\rl_agent\candidate_weights.pt") "Initial agent weights"
$NodeExe = Join-Path $NodeBin "node.exe"
if (Test-Path -LiteralPath $NodeExe) {
    $env:LUX_NODE_BINARY = $NodeExe
}
$env:Path = "$NodeBin;$env:Path"
$env:WANDB_MODE = "offline"
New-Item -ItemType Directory -Path $LeagueRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $BestAgent)) {
    Copy-Item -LiteralPath $InitialAgentPath -Destination $BestAgent -Recurse
}
if (-not (Test-Path -LiteralPath $LearnerAgent)) {
    Copy-Item -LiteralPath $InitialAgentPath -Destination $LearnerAgent -Recurse
}

$Completed = $StartAtGames
while ($Completed -lt $TotalGames) {
    $ThisStageGames = [Math]::Min($GamesPerStage, $TotalGames - $Completed)
    $Completed += $ThisStageGames
    $StageName = "game_stage_{0:D5}" -f $Completed
    $StageDir = Join-Path $LeagueRoot $StageName
    $CandidateAgent = Join-Path $StageDir "candidate_agent"
    $ReplayDir = Join-Path $StageDir "replays"
    New-Item -ItemType Directory -Path $StageDir, $ReplayDir -Force | Out-Null

    if (-not $EvalOnly) {
        $LoadDir = Join-Path $LearnerAgent "lux_ai\rl_agent"
        $LoadFile = "candidate_weights.pt"
        Write-Host "Training $StageName ($ThisStageGames games; balanced maps 12/16/24/32)..."
        Push-Location $StageDir
        try {
            $TrainArgs = @(
                "--config-name=$ConfigName",
                "total_games=$ThisStageGames",
                "total_steps=$($ThisStageGames * 400)",
                "checkpoint_freq=0",
                "teacher_bc_game_offset=$($Completed - $ThisStageGames)",
                "load_dir=$($LoadDir.Replace('\','/'))",
                "checkpoint_file=$LoadFile",
                "weights_only=true",
                "hydra.run.dir=$($StageDir.Replace('\','/'))"
            ) + $TrainOverrides
            & $Python (Join-Path $ProjectRoot "run_monobeast.py") @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                $ExistingWeights = Get-ChildItem -LiteralPath $StageDir -Filter "*_weights.pt" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($ExistingWeights) {
                    Write-Warning "Training process exited with code $LASTEXITCODE in $StageName, but checkpoint exists: $($ExistingWeights.Name). Continuing."
                } else {
                    throw "Training failed in $StageName"
                }
            }
        } finally { Pop-Location }
        $Weights = Get-ChildItem -LiteralPath $StageDir -Filter "*_weights.pt" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $Weights) { throw "No weights checkpoint found in $StageDir" }
        $ResolvedModelConfig = Join-Path $StageDir "config.yaml"
        Copy-Agent $LearnerAgent $CandidateAgent $Weights.FullName $ResolvedModelConfig
    }

    if ($TrainOnly) {
        if ($AccumulateShadowLearner) {
            Write-Host "CHECKPOINT: advancing shadow learner in train-only mode."
            if (Test-Path -LiteralPath $LearnerAgent) {
                Remove-Item -LiteralPath $LearnerAgent -Recurse -Force
            }
            Copy-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent -Recurse
        }
        continue
    }
    $ShouldEvaluate = (
        -not $AccumulateShadowLearner -or
        $Completed % $EvaluationEveryGames -eq 0 -or
        $Completed -eq $TotalGames
    )
    if (-not $ShouldEvaluate) {
        Write-Host "CHECKPOINT: advancing shadow learner without full evaluation."
        if (Test-Path -LiteralPath $LearnerAgent) {
            Remove-Item -LiteralPath $LearnerAgent -Recurse -Force
        }
        Copy-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent -Recurse
        continue
    }
    Assert-Path (Join-Path $CandidateAgent "main.py") "Candidate agent"
    $ReplayFiles = @()
    $ReplayFailures = @()
    $StageSeeds = Resolve-EvaluationSeeds $Seeds $RandomSeedCount
    $SeedTextPath = Join-Path $StageDir "evaluation_seeds.txt"
    $SeedJsonPath = Join-Path $StageDir "evaluation_seeds.json"
    [IO.File]::WriteAllText($SeedTextPath, ($StageSeeds -join "`r`n") + "`r`n", [Text.UTF8Encoding]::new($false))
    $SeedRecord = [PSCustomObject]@{
        generated_at = (Get-Date).ToString("o")
        random_seed_count = $RandomSeedCount
        seeds = @($StageSeeds)
        map_sizes = @($EvaluationMapSizes)
    }
    [IO.File]::WriteAllText($SeedJsonPath, ($SeedRecord | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
    Write-Host "Evaluation seeds: $($StageSeeds -join ', ')"
    foreach ($MapSize in $EvaluationMapSizes) {
        foreach ($Seed in $StageSeeds) {
            foreach ($Opponent in $Opponents) {
                $OpponentName = $Opponent[0]
                $OpponentMain = Join-Path $Opponent[1] "main.py"
                $CandidateMain = Join-Path $CandidateAgent "main.py"
                $Games = @(
                    @($CandidateMain, $OpponentMain, "map_${MapSize}x${MapSize}_vs_${OpponentName}_${Seed}_p0.json"),
                    @($OpponentMain, $CandidateMain, "map_${MapSize}x${MapSize}_vs_${OpponentName}_${Seed}_p1.json")
                )
                foreach ($Game in $Games) {
                    $Output = Join-Path $ReplayDir $Game[2]
                    Write-Host "Replay $($Game[2])"
                    try {
                        Invoke-Game $Game[0] $Game[1] $Seed $MapSize $Output
                        if (Test-Replay $Output) {
                            $ReplayFiles += $Output
                        }
                    } catch {
                        if (-not $ContinueOnReplayFailure) { throw }
                        Write-Warning "Skipping failed replay $($Game[2]): $($_.Exception.Message)"
                        $ReplayFailures += [PSCustomObject]@{
                            map_size = $MapSize
                            seed = $Seed
                            opponent = $OpponentName
                            file = $Game[2]
                            error = $_.Exception.Message
                        }
                    }
                }
            }
        }
    }

    $FailuresPath = Join-Path $StageDir "replay_failures.json"
    [IO.File]::WriteAllText($FailuresPath, ($ReplayFailures | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
    if ($ReplayFiles.Count -eq 0) {
        throw "No completed replays. Failures were written to: $FailuresPath"
    }

    $SummaryPath = Join-Path $StageDir "promotion.json"
    & $Python (Join-Path $PSScriptRoot "evaluate_generalization_promotion.py") `
        @ReplayFiles --output $SummaryPath | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Promotion evaluation failed in $StageName" }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if ($Summary.promote) {
        Write-Host "PROMOTED: $StageName becomes the V8 generalization best."
        if (Test-Path -LiteralPath $BestAgent) { Remove-Item -LiteralPath $BestAgent -Recurse -Force }
        Copy-Item -LiteralPath $CandidateAgent -Destination $BestAgent -Recurse
        if (Test-Path -LiteralPath $LearnerAgent) { Remove-Item -LiteralPath $LearnerAgent -Recurse -Force }
        if ($AccumulateShadowLearner) {
            Copy-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent -Recurse
        } else {
            Move-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent
        }
    } elseif ($AccumulateShadowLearner -and $Summary.shadow_safe) {
        Write-Host "REJECTED FOR CHAMPION: candidate remains the cumulative shadow learner."
        if (Test-Path -LiteralPath $LearnerAgent) {
            Remove-Item -LiteralPath $LearnerAgent -Recurse -Force
        }
        Copy-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent -Recurse
        $Summary.failed_checks | Format-Table name, value, limit -AutoSize
    } else {
        Write-Host "REJECTED: preserving the previous best and safe learner."
        $Summary.failed_checks | Format-Table name, value, limit -AutoSize
        if ($AccumulateShadowLearner -and -not $Summary.shadow_safe) {
            Write-Host "CATASTROPHIC SHADOW REGRESSION: rolling back to the previous safe learner."
            $Summary.shadow_failed_checks | Format-Table name, value, limit -AutoSize
        }
    }
    $CurrentStage400Hash = (Get-FileHash -LiteralPath $Stage400Weights -Algorithm SHA256).Hash
    if ($CurrentStage400Hash -ne $Stage400Hash) {
        throw "Immutable stage 400 baseline changed during the league run"
    }
}

Write-Host "League finished. Immutable baseline: $Stage400Agent"
Write-Host "V8 best: $BestAgent"
