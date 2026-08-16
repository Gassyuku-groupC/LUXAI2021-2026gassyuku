[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateAgent,
    [string]$OutputDir = "outputs\imitation_bc_hq_v1\evaluation",
    [string[]]$Seeds = @(),
    [int]$RandomSeedCount = 4,
    [string[]]$EvaluationMapSizes = @("12", "16"),
    [int]$ReplayTimeoutSeconds = 300,
    [int]$Map32ReplayTimeoutSeconds = 0,
    [int]$FirstOpponentReplayTimeoutSeconds = 0,
    [int]$MaxReplayAttempts = 2,
    [string[]]$OpponentNames = @("first", "stage400", "v4_stage350"),
    [switch]$ContinueOnReplayFailure,
    [switch]$AllowNoCompletedReplays,
    [int]$ReplayHeartbeatSeconds = 0,
    [switch]$VerboseReplayProcess
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LuxCli = Join-Path $ProjectRoot "node_modules\.bin\lux-ai-2021.CMD"
$LuxCliJs = Join-Path $ProjectRoot "node_modules\@lux-ai\2021-challenge\lib\es5\bin\index.js"
$ReplayConverter = Join-Path $PSScriptRoot "convert_replay_stateful.js"
$NodeBin = "C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
$NodeExe = Join-Path $NodeBin "node.exe"
$CurrentPath = [System.Environment]::GetEnvironmentVariable("Path", "Process")
if ([string]::IsNullOrEmpty($CurrentPath)) {
    $CurrentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Process")
}
[System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[System.Environment]::SetEnvironmentVariable("Path", "$NodeBin;$CurrentPath", "Process")
$LuxPackageRoot = Join-Path $ProjectRoot "node_modules\@lux-ai\2021-challenge"
$PnpmRoot = Join-Path $ProjectRoot "node_modules\.pnpm"
$NodePathParts = @(
    (Join-Path $LuxPackageRoot "node_modules"),
    (Join-Path $ProjectRoot "node_modules"),
    (Join-Path $PnpmRoot "node_modules")
)
$ExistingNodePath = [System.Environment]::GetEnvironmentVariable("NODE_PATH", "Process")
if (-not [string]::IsNullOrEmpty($ExistingNodePath)) {
    $NodePathParts += $ExistingNodePath
}
[System.Environment]::SetEnvironmentVariable("NODE_PATH", ($NodePathParts -join ";"), "Process")

if (-not [IO.Path]::IsPathRooted($CandidateAgent)) {
    $CandidateAgent = Join-Path $ProjectRoot $CandidateAgent
}
if (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}

function Convert-IntList([string[]]$Values, [string]$Name) {
    $Items = @()
    foreach ($Value in $Values) {
        foreach ($Part in ($Value -split ",")) {
            $Trimmed = $Part.Trim()
            if ($Trimmed.Length -eq 0) { continue }
            $Parsed = 0
            if (-not [int]::TryParse($Trimmed, [ref]$Parsed)) {
                throw "Invalid integer in ${Name}: '$Trimmed'"
            }
            $Items += $Parsed
        }
    }
    return $Items
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

$Seeds = Convert-IntList $Seeds "Seeds"
$Seeds = Resolve-EvaluationSeeds $Seeds $RandomSeedCount
$EvaluationMapSizes = Convert-IntList $EvaluationMapSizes "EvaluationMapSizes"
foreach ($MapSize in $EvaluationMapSizes) {
    if ($MapSize -notin @(12, 16, 24, 32)) {
        throw "Unsupported map size: $MapSize. Expected one of 12, 16, 24, 32."
    }
}

$OpponentPool = @(
    @("first", (Join-Path $ProjectRoot "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research")),
    @("stage400", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v7_16x16\best_agent")),
    @("v4_stage350", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v4_16x16\learner_agent")),
    @("v5_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v5_16x16\best_agent")),
    @("v6_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v6_16x16\best_agent")),
    @("v7_refine_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v7_refine_16x16\best_agent")),
    @("v8_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v8_generalization\best_agent")),
    @("v9_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v9_robustness\best_agent")),
    @("v11_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v11_stability\best_agent")),
    @("v11b_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v11b_from_v10\best_agent")),
    @("v12_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v12_strategy_buffer\best_agent")),
    @("v12b_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v12b_side_balance_risk_size\best_agent")),
    @("v12c_best", (Join-Path $ProjectRoot "outputs\auto_league_dagger_v12c_side_balance_delayed_risk_size\best_agent")),
    @("public_huikang_private", (Join-Path $ProjectRoot "external_agents\kaggle_public\huikang_private\template")),
    @("public_huikang_agent_b", (Join-Path $ProjectRoot "external_agents\kaggle_public\huikang_evaluation\agent-b")),
    @("public_ilialar_risk_averse", (Join-Path $ProjectRoot "external_agents\kaggle_public\ilialar_risk_averse")),
    @("public_dwight_q_learning", (Join-Path $ProjectRoot "external_agents\kaggle_public\dwight_q_learning")),
    @("public_arpit_rule_based", (Join-Path $ProjectRoot "external_agents\kaggle_public\arpit_rule_based")),
    @("group_4th_0814_imitate_2th", (Join-Path $ProjectRoot "outputs\4th-0814-imitate-2th"))
)
$OpponentNameSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($Name in $OpponentNames) {
    foreach ($Part in ($Name -split ",")) {
        $Trimmed = $Part.Trim()
        if ($Trimmed.Length -gt 0) { [void]$OpponentNameSet.Add($Trimmed) }
    }
}
$Opponents = @($OpponentPool | Where-Object { $OpponentNameSet.Contains($_[0]) })
if ($Opponents.Count -eq 0) {
    $AvailableOpponents = ($OpponentPool | ForEach-Object { $_[0] }) -join ", "
    throw "No opponents selected. Expected any of: $AvailableOpponents."
}

function Assert-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label was not found: $Path" }
}

function Test-Replay([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        $Replay = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $null -ne $Replay -and (Get-Item -LiteralPath $Path).Length -gt 0
    } catch { return $false }
}

function Test-EmptyFile([string]$Path) {
    return (-not (Test-Path -LiteralPath $Path)) -or (Get-Item -LiteralPath $Path).Length -eq 0
}

function Start-CleanProcess(
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$StdoutLog,
    [string]$StderrLog
) {
    $CleanArguments = @()
    foreach ($Argument in $ArgumentList) {
        $CleanArgument = [string]$Argument
        if ($CleanArgument.StartsWith('"') -and $CleanArgument.EndsWith('"')) {
            $CleanArgument = $CleanArgument.Substring(1, $CleanArgument.Length - 2)
        }
        $CleanArguments += $CleanArgument
    }
    if ($VerboseReplayProcess) {
        Write-Host "Launching process: $FilePath"
        Write-Host "stdout: $StdoutLog"
        Write-Host "stderr: $StderrLog"
    }
    return Start-Process -FilePath $FilePath -ArgumentList $CleanArguments -PassThru `
        -NoNewWindow -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
}

function Stop-ProcessTreeQuietly([int]$ProcessId) {
    try {
        & cmd.exe /d /s /c "taskkill /PID $ProcessId /T /F >NUL 2>NUL" | Out-Null
    } catch {
    }
}

$script:CurrentReplayProcessId = $null
trap {
    if ($null -ne $script:CurrentReplayProcessId) {
        Stop-ProcessTreeQuietly $script:CurrentReplayProcessId
        $script:CurrentReplayProcessId = $null
    }
    throw
}

function Get-ProcessTreeSummary([int]$RootProcessId) {
    try {
        $All = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        $ChildrenByParent = @{}
        foreach ($Proc in $All) {
            if (-not $ChildrenByParent.ContainsKey($Proc.ParentProcessId)) {
                $ChildrenByParent[$Proc.ParentProcessId] = @()
            }
            $ChildrenByParent[$Proc.ParentProcessId] += $Proc
        }
        $Queue = @($RootProcessId)
        $Seen = [System.Collections.Generic.HashSet[int]]::new()
        $Rows = @()
        while ($Queue.Count -gt 0) {
            $Pid = [int]$Queue[0]
            $Queue = @($Queue | Select-Object -Skip 1)
            if (-not $Seen.Add($Pid)) { continue }
            $Proc = $All | Where-Object { $_.ProcessId -eq $Pid } | Select-Object -First 1
            if ($null -ne $Proc) {
                $Cpu = ""
                try {
                    $Live = Get-Process -Id $Pid -ErrorAction Stop
                    $Cpu = "{0:n1}s" -f $Live.CPU
                } catch {
                    $Cpu = "?"
                }
                $Rows += ("{0}:{1}:cpu={2}" -f $Proc.Name, $Pid, $Cpu)
            }
            foreach ($Child in @($ChildrenByParent[$Pid])) {
                $Queue += [int]$Child.ProcessId
            }
        }
        return ($Rows -join " | ")
    } catch {
        return "process_tree_unavailable"
    }
}

function Invoke-Game(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [int]$MapSize,
    [string]$Output,
    [int]$TimeoutSeconds
) {
    if (Test-Replay $Output) {
        Write-Host "Replay complete; skipping $([IO.Path]::GetFileName($Output))"
        return
    }
    $CommandReplay = "$Output.commands.json"
    $Arguments = @(
        "`"$Player0`"", "`"$Player1`"", "--python", "`"$Python`"",
        "--seed", $Seed, "--loglevel", "0", "--memory", "8000",
        "--maxtime", "60000", "--width", $MapSize, "--height", $MapSize,
        "--storeLogs=true", "--statefulReplay=false", "--out", "`"$CommandReplay`""
    )
    $StdoutLog = "$CommandReplay.stdout.log"
    $StderrLog = "$CommandReplay.stderr.log"

    for ($Attempt = 1; $Attempt -le $MaxReplayAttempts; $Attempt++) {
        if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output -Force }
        if (Test-Path -LiteralPath $CommandReplay) { Remove-Item -LiteralPath $CommandReplay -Force }
        Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

        if ($Attempt -gt 1) {
            Write-Host "Retry replay attempt $Attempt/$MaxReplayAttempts for $([IO.Path]::GetFileName($Output))"
        }
        $Process = Start-CleanProcess -FilePath $NodeExe -ArgumentList (@($LuxCliJs) + $Arguments) `
            -StdoutLog $StdoutLog -StderrLog $StderrLog
        $script:CurrentReplayProcessId = $Process.Id
        if ($VerboseReplayProcess) {
            Write-Host "Started replay process pid=$($Process.Id)"
        }
        $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        $StartedAt = [DateTime]::UtcNow
        $NextHeartbeat = $StartedAt.AddSeconds($ReplayHeartbeatSeconds)
        while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
            Start-Sleep -Seconds 2
            if (Test-Replay $CommandReplay) { break }
            if ($ReplayHeartbeatSeconds -gt 0 -and [DateTime]::UtcNow -ge $NextHeartbeat) {
                $Elapsed = [int]([DateTime]::UtcNow - $StartedAt).TotalSeconds
                $StdoutBytes = if (Test-Path -LiteralPath $StdoutLog) { (Get-Item -LiteralPath $StdoutLog).Length } else { 0 }
                $StderrBytes = if (Test-Path -LiteralPath $StderrLog) { (Get-Item -LiteralPath $StderrLog).Length } else { 0 }
                $CommandBytes = if (Test-Path -LiteralPath $CommandReplay) { (Get-Item -LiteralPath $CommandReplay).Length } else { 0 }
                $Tree = Get-ProcessTreeSummary $Process.Id
                Write-Host "Replay still running pid=$($Process.Id) elapsed=${Elapsed}s command_bytes=$CommandBytes stdout_bytes=$StdoutBytes stderr_bytes=$StderrBytes tree=[$Tree]"
                $NextHeartbeat = [DateTime]::UtcNow.AddSeconds($ReplayHeartbeatSeconds)
            }
            $OutputText = if (Test-Path -LiteralPath $StdoutLog) {
                Get-Content -LiteralPath $StdoutLog -Raw
            } else { "" }
            if ($OutputText -match 'timed out after|agent [01].*exception') {
                Stop-ProcessTreeQuietly $Process.Id
                $Tail = ($OutputText -split "`r?`n" | Select-Object -Last 12) -join "`n"
                throw "Agent failed during replay: $Output`n$Tail"
            }
        }
        if (Test-Replay $CommandReplay -and -not $Process.HasExited) {
            if (-not $Process.WaitForExit(5000)) {
                Stop-ProcessTreeQuietly $Process.Id
            }
        }
        $script:CurrentReplayProcessId = $null

        if (-not $Process.HasExited -and -not (Test-Replay $CommandReplay)) {
            Stop-ProcessTreeQuietly $Process.Id
            $script:CurrentReplayProcessId = $null
            $StartupLikeFailure = (Test-EmptyFile $CommandReplay) -and (Test-EmptyFile $StdoutLog) -and (Test-EmptyFile $StderrLog)
            if ($StartupLikeFailure -and $Attempt -lt $MaxReplayAttempts) {
                Start-Sleep -Seconds 3
                continue
            }
            if (Test-Path -LiteralPath $CommandReplay) { Remove-Item -LiteralPath $CommandReplay -Force }
            throw "Replay timed out after $TimeoutSeconds seconds: $Output"
        }

        if (-not (Test-Replay $CommandReplay)) {
            if ($Attempt -lt $MaxReplayAttempts) {
                Start-Sleep -Seconds 3
                continue
            }
            throw "Command replay failed: $CommandReplay"
        }

        Write-Host "Converting replay to stateful JSON..."
        $ConvertProcess = Start-CleanProcess -FilePath "node" -ArgumentList @(
            "`"$ReplayConverter`"", "`"$CommandReplay`"", "`"$Output`""
        ) -StdoutLog "$Output.convert.stdout.log" -StderrLog "$Output.convert.stderr.log"
        if (-not $ConvertProcess.WaitForExit(300000)) {
            Stop-ProcessTreeQuietly $ConvertProcess.Id
            throw "Stateful replay conversion timed out: $Output"
        }
        if (Test-Replay $Output) { return }
        if ($Attempt -lt $MaxReplayAttempts) {
            Start-Sleep -Seconds 3
            continue
        }
        $ConvertError = Get-Content "$Output.convert.stderr.log" -Raw -ErrorAction SilentlyContinue
        throw "Stateful replay conversion failed: $Output`n$ConvertError"
    }
}

Assert-Path $Python "Virtual-environment Python"
Assert-Path $NodeExe "Node runtime"
Assert-Path $LuxCliJs "Lux CLI JavaScript entry"
Assert-Path $ReplayConverter "Stateful replay converter"
Assert-Path (Join-Path $CandidateAgent "main.py") "Candidate agent"
Assert-Path (Join-Path $CandidateAgent "lux_ai\rl_agent\candidate_weights.pt") "Candidate weights"
foreach ($Opponent in $Opponents) {
    Assert-Path (Join-Path $Opponent[1] "main.py") ("Opponent {0}" -f $Opponent[0])
}

$ReplayDir = Join-Path $OutputDir "replays"
New-Item -ItemType Directory -Path $OutputDir, $ReplayDir -Force | Out-Null
$SeedTextPath = Join-Path $OutputDir "evaluation_seeds.txt"
$SeedJsonPath = Join-Path $OutputDir "evaluation_seeds.json"
[IO.File]::WriteAllText($SeedTextPath, ($Seeds -join "`r`n") + "`r`n", [Text.UTF8Encoding]::new($false))
$SeedRecord = [PSCustomObject]@{
    generated_at = (Get-Date).ToString("o")
    random_seed_count = $RandomSeedCount
    seeds = @($Seeds)
    map_sizes = @($EvaluationMapSizes)
}
[IO.File]::WriteAllText($SeedJsonPath, ($SeedRecord | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Write-Host "Evaluation seeds: $($Seeds -join ', ')"

$ReplayFiles = @()
$ReplayFailures = @()
foreach ($MapSize in $EvaluationMapSizes) {
    foreach ($Seed in $Seeds) {
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
                $GameTimeoutSeconds = $ReplayTimeoutSeconds
                if ($MapSize -eq 32 -and $Map32ReplayTimeoutSeconds -gt 0) {
                    $GameTimeoutSeconds = [Math]::Max($GameTimeoutSeconds, $Map32ReplayTimeoutSeconds)
                }
                if ($OpponentName -eq "first" -and $FirstOpponentReplayTimeoutSeconds -gt 0) {
                    $GameTimeoutSeconds = [Math]::Max($GameTimeoutSeconds, $FirstOpponentReplayTimeoutSeconds)
                }
                Write-Host "Replay $($Game[2])"
                try {
                    Invoke-Game $Game[0] $Game[1] $Seed $MapSize $Output $GameTimeoutSeconds
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

$FailuresPath = Join-Path $OutputDir "replay_failures.json"
[IO.File]::WriteAllText($FailuresPath, ($ReplayFailures | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
if ($ReplayFiles.Count -eq 0) {
    if ($AllowNoCompletedReplays) {
        Write-Warning "No completed replays. Failures were written to: $FailuresPath"
        exit 0
    }
    throw "No completed replays. Failures were written to: $FailuresPath"
}

$SummaryPath = Join-Path $OutputDir "promotion_metrics.json"
& $Python (Join-Path $PSScriptRoot "evaluate_generalization_promotion.py") `
    @ReplayFiles --output $SummaryPath | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Generalization evaluation failed." }
Write-Host "Evaluation summary: $SummaryPath"
