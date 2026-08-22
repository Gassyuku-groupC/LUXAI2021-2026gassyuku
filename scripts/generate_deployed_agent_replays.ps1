[CmdletBinding()]
param(
    [string]$CurrentAgent = "outputs\current_agent",
    [string]$BestAgent = "outputs\submission_packages\best_agent",
    [string]$FirstAgent = "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research",
    [string]$Stage400Agent = "outputs\auto_league_dagger_v7_16x16\best_agent",
    [string]$Stage350Agent = "outputs\auto_league_dagger_v4_16x16\learner_agent",
    [string]$NodeExe = "",
    [int[]]$Seeds = @(20260821),
    [int[]]$MapSizes = @(12, 24),
    [string[]]$OpponentNames = @("best_agent", "first", "stage350", "stage400"),
    [int[]]$Sides = @(0, 1),
    [string]$OutputDir = "outputs\spatial_risk_deployed_replays",
    [int]$AgentTurnTimeoutMs = 30000,
    [int]$TimeoutSeconds = 420,
    [int]$HeartbeatSeconds = 60,
    [switch]$ContinueOnFailure
)

$ErrorActionPreference = "Stop"
$LuxRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $LuxRoot ".venv\Scripts\python.exe"
$PythonDir = Split-Path -Parent $Python
if (-not $NodeExe) {
    $bundledNode = Join-Path $LuxRoot ".tools\node16\node_modules\node\bin\node.exe"
    if (Test-Path -LiteralPath $bundledNode) {
        $NodeExe = $bundledNode
    } else {
        $nodeCommand = Get-Command node.exe -ErrorAction Stop
        $NodeExe = $nodeCommand.Source
    }
}
$Node = (Resolve-Path -LiteralPath $NodeExe).Path
$NodeRoot = Split-Path -Parent $Node
$LuxCli = Join-Path $LuxRoot "node_modules\@lux-ai\2021-challenge\lib\es5\bin\index.js"
$Converter = Join-Path $LuxRoot "scripts\convert_replay_stateful.js"

function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $LuxRoot $Path
}

$CurrentAgent = Resolve-ProjectPath $CurrentAgent
$BestAgent = Resolve-ProjectPath $BestAgent
$FirstAgent = Resolve-ProjectPath $FirstAgent
$Stage400Agent = Resolve-ProjectPath $Stage400Agent
$Stage350Agent = Resolve-ProjectPath $Stage350Agent

if (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $PSScriptRoot) $OutputDir
}
$ReplayDir = Join-Path $OutputDir "replays"
$LogDir = Join-Path $OutputDir "logs"
New-Item -ItemType Directory -Path $ReplayDir, $LogDir -Force | Out-Null

$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", "$NodeRoot;$PythonDir;$CurrentPath", "Process")
[Environment]::SetEnvironmentVariable("VIRTUAL_ENV", (Join-Path $LuxRoot ".venv"), "Process")
$NodeModules = @(
    (Join-Path $LuxRoot "node_modules\@lux-ai\2021-challenge\node_modules"),
    (Join-Path $LuxRoot "node_modules"),
    (Join-Path $LuxRoot "node_modules\.pnpm\node_modules")
)
[Environment]::SetEnvironmentVariable("NODE_PATH", ($NodeModules -join ";"), "Process")

$Opponents = @(
    [pscustomobject]@{ Name = "best_agent"; Path = $BestAgent },
    [pscustomobject]@{ Name = "first"; Path = $FirstAgent },
    [pscustomobject]@{ Name = "stage400"; Path = $Stage400Agent }
)
if ($Stage350Agent) {
    $Opponents += [pscustomobject]@{ Name = "stage350"; Path = $Stage350Agent }
}
$selectedOpponentNames = [System.Collections.Generic.HashSet[string]]::new(
    [string[]]$OpponentNames,
    [System.StringComparer]::OrdinalIgnoreCase
)
$Opponents = @($Opponents | Where-Object { $selectedOpponentNames.Contains($_.Name) })
if ($Opponents.Count -eq 0) { throw "No opponents selected: $($OpponentNames -join ', ')" }
foreach ($side in $Sides) {
    if ($side -notin @(0, 1)) { throw "Invalid side $side; expected 0 or 1." }
}

foreach ($required in @($Python, $Node, $LuxCli, $Converter, (Join-Path $CurrentAgent "main.py"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required path not found: $required" }
}
foreach ($opponent in $Opponents) {
    if (-not (Test-Path -LiteralPath (Join-Path $opponent.Path "main.py"))) {
        throw "Opponent agent not found: $($opponent.Name) at $($opponent.Path)"
    }
}

function Stop-Tree([int]$Id) {
    try { & taskkill.exe /PID $Id /T /F 2>$null | Out-Null } catch { }
}

function Get-DescendantProcessIds([int]$RootId) {
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        return @()
    }
    $descendants = [System.Collections.Generic.List[int]]::new()
    $frontier = [System.Collections.Generic.Queue[int]]::new()
    $frontier.Enqueue($RootId)
    while ($frontier.Count -gt 0) {
        $parentId = $frontier.Dequeue()
        foreach ($child in $processes | Where-Object { $_.ParentProcessId -eq $parentId }) {
            $childId = [int]$child.ProcessId
            $descendants.Add($childId)
            $frontier.Enqueue($childId)
        }
    }
    return @($descendants)
}

function Stop-MatchProcesses([int]$NodeId) {
    $descendants = @(Get-DescendantProcessIds $NodeId)
    [array]::Reverse($descendants)
    foreach ($processId in $descendants) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $NodeId -Force -ErrorAction SilentlyContinue
    Stop-Tree $NodeId
}

function Test-JsonReplay([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -eq 0) { return $false }
    try {
        $replay = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $null -ne $replay -and $null -ne $replay.results
    } catch {
        return $false
    }
}

function Test-AgentTimeout([string]$StdoutPath, [string]$StderrPath) {
    foreach ($path in @($StdoutPath, $StderrPath)) {
        if ((Test-Path -LiteralPath $path) -and
            (Select-String -LiteralPath $path -Pattern "timed out after" -Quiet)) {
            return $true
        }
    }
    return $false
}

function Invoke-Match(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [int]$MapSize,
    [string]$Name
) {
    $commandReplay = Join-Path $ReplayDir "$Name.commands.json"
    $statefulReplay = Join-Path $ReplayDir "$Name.json"
    $stdout = Join-Path $LogDir "$Name.stdout.log"
    $stderr = Join-Path $LogDir "$Name.stderr.log"
    $arguments = @(
        "`"$LuxCli`"", "`"$Player0`"", "`"$Player1`"", "--python", "`"$Python`"",
        "--seed", $Seed, "--loglevel", "1", "--memory", "8000",
        "--maxtime", $AgentTurnTimeoutMs, "--width", $MapSize, "--height", $MapSize,
        "--storeLogs=true", "--statefulReplay=false", "--out", "`"$commandReplay`""
    )
    if (-not (Test-JsonReplay $commandReplay)) {
        Write-Host "Running $Name"
        $process = Start-Process -FilePath $Node -ArgumentList $arguments -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        $started = [DateTime]::UtcNow
        $nextHeartbeat = $started.AddSeconds($HeartbeatSeconds)
        while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-AgentTimeout $stdout $stderr) {
                Stop-MatchProcesses $process.Id
                throw "Agent turn timeout was reported by the Lux engine: $Name"
            }
            if (Test-JsonReplay $commandReplay) { break }
            if ($HeartbeatSeconds -gt 0 -and [DateTime]::UtcNow -ge $nextHeartbeat) {
                $elapsed = [int]([DateTime]::UtcNow - $started).TotalSeconds
                $commandBytes = if (Test-Path -LiteralPath $commandReplay) { (Get-Item -LiteralPath $commandReplay).Length } else { 0 }
                $stdoutBytes = if (Test-Path -LiteralPath $stdout) { (Get-Item -LiteralPath $stdout).Length } else { 0 }
                $stderrBytes = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
                Write-Host "Replay active name=$Name pid=$($process.Id) elapsed=${elapsed}s command_bytes=$commandBytes stdout_bytes=$stdoutBytes stderr_bytes=$stderrBytes"
                $nextHeartbeat = [DateTime]::UtcNow.AddSeconds($HeartbeatSeconds)
            }
        }
        if (Test-JsonReplay $commandReplay) {
            Stop-MatchProcesses $process.Id
        } else {
            Stop-MatchProcesses $process.Id
            throw "Replay did not produce valid JSON within $TimeoutSeconds seconds: $Name"
        }
    } else {
        Write-Host "Reusing completed command replay $Name"
    }
    if (Test-AgentTimeout $stdout $stderr) {
        throw "Agent turn timeout was reported by the Lux engine: $Name"
    }
    if (Test-JsonReplay $statefulReplay) {
        return [pscustomobject]@{
            name = $Name; map_size = $MapSize; seed = $Seed; replay = $statefulReplay;
            command_replay = $commandReplay; bytes = (Get-Item -LiteralPath $statefulReplay).Length
        }
    }
    $convertOut = Join-Path $LogDir "$Name.convert.stdout.log"
    $convertErr = Join-Path $LogDir "$Name.convert.stderr.log"
    $convert = Start-Process -FilePath $Node -ArgumentList @(
        "`"$Converter`"", "`"$commandReplay`"", "`"$statefulReplay`""
    ) `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $convertOut -RedirectStandardError $convertErr
    if (-not $convert.WaitForExit(300000)) {
        Stop-Tree $convert.Id
        throw "Stateful conversion timed out: $Name"
    }
    $conversionDeadline = [DateTime]::UtcNow.AddSeconds(5)
    while (-not (Test-JsonReplay $statefulReplay) -and [DateTime]::UtcNow -lt $conversionDeadline) {
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-JsonReplay $statefulReplay)) {
        throw "Stateful conversion did not produce valid JSON: $Name"
    }
    return [pscustomobject]@{
        name = $Name; map_size = $MapSize; seed = $Seed; replay = $statefulReplay;
        command_replay = $commandReplay; bytes = (Get-Item -LiteralPath $statefulReplay).Length
    }
}

$completed = @()
$failures = @()
foreach ($mapSize in $MapSizes) {
    foreach ($seed in $Seeds) {
        foreach ($opponent in $Opponents) {
            foreach ($side in $Sides) {
                $name = "map_${mapSize}x${mapSize}_vs_$($opponent.Name)_${seed}_p${side}"
                $p0 = if ($side -eq 0) { Join-Path $CurrentAgent "main.py" } else { Join-Path $opponent.Path "main.py" }
                $p1 = if ($side -eq 0) { Join-Path $opponent.Path "main.py" } else { Join-Path $CurrentAgent "main.py" }
                try {
                    $completed += Invoke-Match $p0 $p1 $seed $mapSize $name
                } catch {
                    $failures += [pscustomobject]@{ name = $name; error = $_.Exception.Message }
                    Write-Warning $_.Exception.Message
                    if (-not $ContinueOnFailure) { throw }
                }
            }
        }
    }
}

$manifest = [pscustomobject]@{
    current_agent = $CurrentAgent
    opponents = $Opponents
    seeds = $Seeds
    map_sizes = $MapSizes
    sides = $Sides
    completed = $completed
    failures = $failures
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir "manifest.json") -Encoding utf8
Write-Host "Completed $($completed.Count) replay(s); failures=$($failures.Count); output=$OutputDir"
