[CmdletBinding()]
param(
    [int]$TotalGames = 128,
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

$Arguments = @{
    TotalGames = $TotalGames
    StartAtGames = $StartAtGames
    GamesPerStage = $GamesPerStage
    EvaluationEveryGames = $EvaluationEveryGames
    Seeds = $Seeds
    RandomSeedCount = $RandomSeedCount
    EvaluationMapSizes = $EvaluationMapSizes
    ReplayTimeoutSeconds = $ReplayTimeoutSeconds
    ConfigName = "conv_teacher_bc_dagger_v11_stability"
    LeagueName = "auto_league_dagger_v11_stability"
    AccumulateShadowLearner = $true
}
if ($TrainOnly) { $Arguments.TrainOnly = $true }
if ($EvalOnly) { $Arguments.EvalOnly = $true }

& (Join-Path $PSScriptRoot "auto_train_generalization_v8.ps1") @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
