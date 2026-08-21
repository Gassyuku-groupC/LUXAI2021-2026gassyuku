[CmdletBinding()]
param(
    [int]$TotalGames = 100,
    [int]$StartAtGames = 0,
    [int]$GamesPerStage = 25,
    [int[]]$Seeds = @(12345, 23456, 34567, 45678),
    [int]$MapSize = 16,
    [int]$ReplayTimeoutSeconds = 300,
    [int]$MaxNightCityLoss = 10,
    [double]$MinWinRate = 0.5,
    [double]$MinReferenceWinRate = 0.5,
    [double]$MinMeanCityTiles = 60.0,
    [string]$ConfigName = "conv_teacher_bc_dagger_v2_16x16",
    [string]$InitialAgent = "local_agents/survival_research_buffer2_finetune_16x16_100000",
    [string]$LeagueName = "auto_league_dagger_v2_16x16",
    [string]$ChampionAgent = "outputs/auto_league_16x16/best_agent",
    [string]$ReferenceAgent = "",
    [string]$InitialWeights = "outputs/auto_league_16x16/game_stage_00100/18080_weights.pt",
    [switch]$TrainOnly,
    [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LuxCli = Join-Path $ProjectRoot "node_modules\.bin\lux-ai-2021.CMD"
$FirstAgent = Join-Path $ProjectRoot "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py"
$LeagueRoot = Join-Path $ProjectRoot ("outputs\{0}" -f $LeagueName)
$BestAgent = Join-Path $LeagueRoot "best_agent"
$LearnerAgent = Join-Path $LeagueRoot "learner_agent"
$NodeBin = "C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"

function Assert-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function Copy-Agent([string]$Source, [string]$Destination, [string]$Weights) {
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
    Get-ChildItem -LiteralPath (Join-Path $Destination "lux_ai\rl_agent") `
        -Filter "*.pt" -File |
        Remove-Item -Force
    Copy-Item -LiteralPath $Weights `
        -Destination (Join-Path $Destination "lux_ai\rl_agent\candidate_weights.pt") -Force
    $ConfigPath = Join-Path $Destination "lux_ai\rl_agent\config.yaml"
    (Get-Content -LiteralPath $ConfigPath -Raw) `
        -replace '(?m)^checkpoint_file:\s*\S+', 'checkpoint_file: candidate_weights.pt' |
        Set-Content -LiteralPath $ConfigPath -Encoding utf8
}

function Invoke-Game(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [string]$Output
) {
    if ((Test-Path -LiteralPath $Output) -and (Get-Item -LiteralPath $Output).Length -gt 0) {
        Write-Host "Replay already complete; skipping $([IO.Path]::GetFileName($Output))"
        return
    }
    $Arguments = @(
        "`"$Player0`"", "`"$Player1`"",
        "--python", "`"$Python`"",
        "--seed", $Seed,
        "--loglevel", "0",
        "--memory", "8000",
        "--maxtime", "120000",
        "--width", $MapSize,
        "--height", $MapSize,
        "--storeLogs=true",
        "--statefulReplay=true",
        "--out", "`"$Output`""
    )
    $Process = Start-Process -FilePath $LuxCli -ArgumentList $Arguments `
        -PassThru -NoNewWindow
    if (-not $Process.WaitForExit($ReplayTimeoutSeconds * 1000)) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
        throw "Replay timed out after $ReplayTimeoutSeconds seconds: $Output"
    }
    # Start-Process may not expose a reliable ExitCode when launching a .CMD
    # shim. The stateful replay is the authoritative output; malformed JSON is
    # rejected later by evaluate_replays.py.
    if (-not (Test-Path -LiteralPath $Output) -or
        (Get-Item -LiteralPath $Output).Length -eq 0) {
        throw "Replay failed: $Output"
    }
}

Assert-Path $Python "Virtual-environment Python"
Assert-Path $LuxCli "Lux CLI"
Assert-Path $FirstAgent "First-place agent"
Assert-Path (Join-Path $ProjectRoot $InitialAgent) "Initial agent"
Assert-Path (Join-Path $ProjectRoot $ChampionAgent) "Champion agent"
Assert-Path (Join-Path $ProjectRoot $InitialWeights) "Initial learner weights"
$ReferenceMain = $null
if ($ReferenceAgent) {
    $ReferenceMain = Join-Path (Join-Path $ProjectRoot $ReferenceAgent) "main.py"
    Assert-Path $ReferenceMain "Reference agent"
}
New-Item -ItemType Directory -Path $LeagueRoot -Force | Out-Null
$env:PATH = "$NodeBin;$env:PATH"
$env:WANDB_MODE = "offline"

if (-not (Test-Path -LiteralPath $BestAgent)) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $ChampionAgent) `
        -Destination $BestAgent -Recurse
}
if (-not (Test-Path -LiteralPath $LearnerAgent)) {
    $ResumeCandidate = Join-Path $LeagueRoot (
        "game_stage_{0:D5}\candidate_agent" -f $StartAtGames
    )
    $LearnerSource = if ($StartAtGames -gt 0 -and
        (Test-Path -LiteralPath $ResumeCandidate)) {
        $ResumeCandidate
    } else {
        $null
    }
    if ($null -ne $LearnerSource) {
        Copy-Item -LiteralPath $LearnerSource -Destination $LearnerAgent -Recurse
    } else {
        Copy-Agent (Join-Path $ProjectRoot $InitialAgent) $LearnerAgent `
            (Join-Path $ProjectRoot $InitialWeights)
    }
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
        $LoadFile = if (Test-Path (Join-Path $LoadDir "candidate_weights.pt")) {
            "candidate_weights.pt"
        } else {
            "100000_weights.pt"
        }
        $EstimatedSteps = $ThisStageGames * 400
        Write-Host "Training $StageName ($ThisStageGames completed games) from cumulative learner..."
        Push-Location $StageDir
        try {
            & $Python (Join-Path $ProjectRoot "run_monobeast.py") `
                "--config-name=$ConfigName" `
                "total_games=$ThisStageGames" `
                "total_steps=$EstimatedSteps" `
                "checkpoint_freq=0" `
                "teacher_bc_game_offset=$($Completed - $ThisStageGames)" `
                "load_dir=$($LoadDir.Replace('\','/'))" `
                "checkpoint_file=$LoadFile" `
                "weights_only=true" `
                "hydra.run.dir=$($StageDir.Replace('\','/'))"
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed in $StageName"
            }
        } finally {
            Pop-Location
        }
        $Weights = Get-ChildItem -LiteralPath $StageDir -Filter "*_weights.pt" |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if (-not $Weights) {
            throw "No weights checkpoint found in $StageDir"
        }
        Copy-Agent $LearnerAgent $CandidateAgent $Weights.FullName
    }

    if ($TrainOnly) {
        continue
    }
    Assert-Path (Join-Path $CandidateAgent "main.py") "Candidate agent"
    $CandidateMain = Join-Path $CandidateAgent "main.py"
    $BestMain = Join-Path $BestAgent "main.py"
    $ReplayFiles = @()
    foreach ($Seed in $Seeds) {
        $Games = @(
            @($CandidateMain, $BestMain, "candidate_vs_best_${Seed}_p0.json"),
            @($BestMain, $CandidateMain, "candidate_vs_best_${Seed}_p1.json"),
            @($CandidateMain, $FirstAgent, "candidate_vs_first_${Seed}_p0.json"),
            @($FirstAgent, $CandidateMain, "candidate_vs_first_${Seed}_p1.json")
        )
        if ($ReferenceMain) {
            $Games += ,@(
                $CandidateMain,
                $ReferenceMain,
                "candidate_vs_reference_${Seed}_p0.json"
            )
            $Games += ,@(
                $ReferenceMain,
                $CandidateMain,
                "candidate_vs_reference_${Seed}_p1.json"
            )
        }
        foreach ($Game in $Games) {
            $Output = Join-Path $ReplayDir $Game[2]
            Write-Host "Replay $($Game[2])"
            Invoke-Game $Game[0] $Game[1] $Seed $Output
            $ReplayFiles += $Output
        }
    }
    $SummaryPath = Join-Path $StageDir "evaluation.json"
    & $Python (Join-Path $PSScriptRoot "evaluate_replays.py") `
        @ReplayFiles --output $SummaryPath | Out-Host
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $ReferenceDetails = @(
        $Summary.details | Where-Object {
            [IO.Path]::GetFileName($_.file) -like "candidate_vs_reference_*"
        }
    )
    $ReferenceWinRate = if ($ReferenceDetails.Count -gt 0) {
        @($ReferenceDetails | Where-Object { $_.rank -eq 1 }).Count / $ReferenceDetails.Count
    } else {
        1.0
    }
    $Promote = (
        $Summary.survival_rate -eq 1.0 -and
        $Summary.worst_night_city_loss -le $MaxNightCityLoss -and
        $Summary.win_rate -ge $MinWinRate -and
        $Summary.mean_city_tiles -ge $MinMeanCityTiles -and
        $ReferenceWinRate -ge $MinReferenceWinRate
    )
    Write-Host (
        "Promotion checks: survival={0:P0}, worst_loss={1}, win_rate={2:P1}, " +
        "mean_tiles={3:N2}, reference_win_rate={4:P1}" -f
        $Summary.survival_rate,
        $Summary.worst_night_city_loss,
        $Summary.win_rate,
        $Summary.mean_city_tiles,
        $ReferenceWinRate
    )
    if ($Promote) {
        Write-Host "PROMOTED: $StageName becomes the new best."
        $PreviousBest = Join-Path $StageDir "previous_best"
        Move-Item -LiteralPath $BestAgent -Destination $PreviousBest
        Copy-Item -LiteralPath $CandidateAgent -Destination $BestAgent -Recurse
        if (Test-Path -LiteralPath $LearnerAgent) {
            Remove-Item -LiteralPath $LearnerAgent -Recurse -Force
        }
        Move-Item -LiteralPath $CandidateAgent -Destination $LearnerAgent
    } else {
        Write-Host "REJECTED: keeping the previous best and learner."
    }
}

Write-Host "League finished."
Write-Host "Best evaluated agent: $BestAgent"
Write-Host "Latest cumulative learner: $LearnerAgent"
