[CmdletBinding()]
param(
    [int]$TotalGames = 16,
    [int]$StartAtGames = 0,
    [int]$GamesPerStage = 16,
    [int]$EvaluationEveryGames = 16,
    [string[]]$Seeds = @(),
    [int]$RandomSeedCount = 4,
    [int[]]$EvaluationMapSizes = @(16),
    [int]$ReplayTimeoutSeconds = 300,
    [int]$RolloutQueueTimeoutSeconds = 180,
    [int]$TrainingStallTimeoutSeconds = 240,
    [switch]$TrainOnly,
    [switch]$EvalOnly,
    [switch]$ContinueOnReplayFailure
)

if ($EvaluationEveryGames -lt $GamesPerStage -or
    $EvaluationEveryGames % $GamesPerStage -ne 0) {
    throw "EvaluationEveryGames must be a multiple of GamesPerStage."
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

$ResolvedSeeds = Convert-IntList $Seeds "Seeds"

$Arguments = @{
    TotalGames = $TotalGames
    StartAtGames = $StartAtGames
    GamesPerStage = $GamesPerStage
    EvaluationEveryGames = $EvaluationEveryGames
    Seeds = $ResolvedSeeds
    RandomSeedCount = $RandomSeedCount
    EvaluationMapSizes = $EvaluationMapSizes
    ReplayTimeoutSeconds = $ReplayTimeoutSeconds
    ConfigName = "conv_teacher_bc_dagger_v15b_aux_loss20_soft"
    LeagueName = "auto_league_aux_risk_v15b_from_best"
    InitialAgent = "outputs\auto_league_dagger_v10_shadow\best_agent"
    AccumulateShadowLearner = $true
    TrainOverrides = @(
        "rollout_queue_timeout_seconds=$RolloutQueueTimeoutSeconds",
        "training_stall_timeout_seconds=$TrainingStallTimeoutSeconds"
    )
}
if ($TrainOnly) { $Arguments.TrainOnly = $true }
if ($EvalOnly) { $Arguments.EvalOnly = $true }
if ($ContinueOnReplayFailure) { $Arguments.ContinueOnReplayFailure = $true }

& (Join-Path $PSScriptRoot "auto_train_generalization_v8.ps1") @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
