[CmdletBinding()]
param(
    [int]$TotalGames = 64,
    [int]$StartAtGames = 0,
    [int]$GamesPerStage = 16,
    [int]$EvaluationEveryGames = 32,
    [int[]]$Seeds = @(),
    [int]$RandomSeedCount = 4,
    [int[]]$EvaluationMapSizes = @(12, 16),
    [int]$ReplayTimeoutSeconds = 300,
    [switch]$TrainOnly,
    [switch]$EvalOnly
)

if ($EvaluationEveryGames -lt $GamesPerStage -or
    $EvaluationEveryGames % $GamesPerStage -ne 0) {
    throw "EvaluationEveryGames must be a multiple of GamesPerStage."
}

function Normalize-ProcessPath {
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $CurrentPath) {
        $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $CurrentPath, "Process")
}

Normalize-ProcessPath

$NodeCommand = Get-Command node -ErrorAction SilentlyContinue
if (-not $NodeCommand) {
    $BundledNode = "C:\Users\YE ZIHAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
    if (Test-Path -LiteralPath $BundledNode) {
        $env:Path = "$(Split-Path -Parent $BundledNode);$env:Path"
        $NodeCommand = Get-Command node -ErrorAction SilentlyContinue
    }
}
if (-not $NodeCommand) {
    throw "Node.js is required by kaggle_environments Lux dimensions, but node was not found on PATH."
}
$env:LUX_NODE_BINARY = $NodeCommand.Source
$env:Path = "$(Split-Path -Parent $NodeCommand.Source);$env:Path"

$Arguments = @{
    TotalGames = $TotalGames
    StartAtGames = $StartAtGames
    GamesPerStage = $GamesPerStage
    EvaluationEveryGames = $EvaluationEveryGames
    Seeds = $Seeds
    RandomSeedCount = $RandomSeedCount
    EvaluationMapSizes = $EvaluationMapSizes
    ReplayTimeoutSeconds = $ReplayTimeoutSeconds
    ConfigName = "conv_teacher_bc_dagger_v12_strategy_buffer"
    LeagueName = "auto_league_dagger_v12_strategy_buffer"
    AccumulateShadowLearner = $true
}
if ($TrainOnly) { $Arguments.TrainOnly = $true }
if ($EvalOnly) { $Arguments.EvalOnly = $true }

& (Join-Path $PSScriptRoot "auto_train_generalization_v8.ps1") @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
