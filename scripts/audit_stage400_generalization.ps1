[CmdletBinding()]
param(
    [int[]]$MapSizes = @(12, 16, 24, 32),
    [int[]]$Seeds = @(12345, 23456, 34567, 45678),
    [int]$ReplayTimeoutSeconds = 300,
    [string]$Agent = "outputs/auto_league_dagger_v7_16x16/best_agent",
    [string]$OutputName = "v7_stage400_generalization"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LuxCli = Join-Path $ProjectRoot "node_modules\.bin\lux-ai-2021.CMD"
$NodeBin = "C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
$OutputRoot = Join-Path $ProjectRoot ("outputs\audits\{0}" -f $OutputName)
$AgentMain = Join-Path (Join-Path $ProjectRoot $Agent) "main.py"

$Opponents = @(
    @(
        "first",
        (Join-Path $ProjectRoot "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research\main.py")
    ),
    @(
        "v4_stage350",
        (Join-Path $ProjectRoot "outputs\auto_league_dagger_v4_16x16\learner_agent\main.py")
    ),
    @(
        "public_working_title",
        (Join-Path $ProjectRoot "internal_testing\public_agents\working_title_bot_tong_hui_kang\main.py")
    )
)

function Assert-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function Invoke-AuditGame(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [int]$MapSize,
    [string]$Output
) {
    if (Test-Path -LiteralPath $Output) {
        try {
            $Replay = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
            if ($null -ne $Replay -and (Get-Item -LiteralPath $Output).Length -gt 0) {
                Write-Host "Replay already complete; skipping $([IO.Path]::GetFileName($Output))"
                return
            }
        }
        catch {
            # A timeout can leave an empty or truncated replay behind.
        }
        Remove-Item -LiteralPath $Output -Force
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
    $Process = Start-Process -FilePath $LuxCli -ArgumentList $Arguments -PassThru -NoNewWindow
    if (-not $Process.WaitForExit($ReplayTimeoutSeconds * 1000)) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
        throw "Replay timed out after $ReplayTimeoutSeconds seconds: $Output"
    }
    if (-not (Test-Path -LiteralPath $Output) -or (Get-Item -LiteralPath $Output).Length -eq 0) {
        throw "Replay failed: $Output"
    }
}

Assert-Path $Python "Virtual-environment Python"
Assert-Path $LuxCli "Lux CLI"
Assert-Path $AgentMain "Audit agent"
foreach ($Opponent in $Opponents) {
    Assert-Path $Opponent[1] ("Opponent {0}" -f $Opponent[0])
}

$env:PATH = "$NodeBin;$env:PATH"
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$AllReplayFiles = @()

foreach ($MapSize in $MapSizes) {
    $MapRoot = Join-Path $OutputRoot ("map_{0}x{0}" -f $MapSize)
    New-Item -ItemType Directory -Path $MapRoot -Force | Out-Null
    $MapReplayFiles = @()
    foreach ($Seed in $Seeds) {
        foreach ($Opponent in $Opponents) {
            $OpponentName = $Opponent[0]
            $OpponentMain = $Opponent[1]
            $Games = @(
                @($AgentMain, $OpponentMain, "${OpponentName}_${Seed}_p0.json"),
                @($OpponentMain, $AgentMain, "${OpponentName}_${Seed}_p1.json")
            )
            foreach ($Game in $Games) {
                $Output = Join-Path $MapRoot $Game[2]
                Write-Host "Map ${MapSize} | Replay $($Game[2])"
                Invoke-AuditGame $Game[0] $Game[1] $Seed $MapSize $Output
                $MapReplayFiles += $Output
                $AllReplayFiles += $Output
            }
        }
    }
    & $Python (Join-Path $PSScriptRoot "evaluate_replays.py") `
        @MapReplayFiles --output (Join-Path $MapRoot "evaluation.json") | Out-Null
}

& $Python (Join-Path $PSScriptRoot "evaluate_replays.py") `
    @AllReplayFiles --output (Join-Path $OutputRoot "evaluation_overall.json") | Out-Null

Write-Host "Audit finished: $OutputRoot"
Write-Host "Replays: $($AllReplayFiles.Count)"
