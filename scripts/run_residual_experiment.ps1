[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentName,

    [string]$TrainScript = "scripts\train_spatial_residual_head.py",
    [string]$ShardsDir = "dataset\processed\imitation_shards_counterfactual_v4_residual",
    [string]$BaseAgent = "outputs\auto_league_dagger_v10_shadow\best_agent",
    [string]$OutputRoot = "outputs\experiments",

    [int]$Epochs = 1,
    [int]$BatchSize = 8,
    [double]$LearningRate = 0.0003,
    [double]$Gamma = 0.05,
    [double]$MaxDelta = 0.75,
    [int]$HiddenChannels = 64,
    [int]$KernelSize = 3,
    [double]$P0ResidualScale = 1.0,
    [double]$P1ResidualScale = 1.0,
    [double]$P0CeMult = 1.0,
    [double]$P1CeMult = 1.0,
    [double]$P0AnchorMult = 1.0,
    [double]$P1AnchorMult = 1.0,
    [double]$KlBeta = 0.2,
    [double]$L2Beta = 0.02,
    [double]$AnchorWeight = 0.05,
    [switch]$StructuredCriticalLoss,
    [double]$SupportWorkerMult = 1.5,
    [double]$SupportCityMult = 0.25,
    [double]$LateWorkerMult = 0.75,
    [double]$LateCityMult = 1.25,
    [double]$SafeCityMult = 1.15,
    [int]$Seed = 20260813,

    [string[]]$EvaluationSeeds = @(
        "471851002", "1064830286", "151752041", "805766398",
        "973956985", "1600245205", "1009133967", "1025747501",
        "852579075", "58176888", "257199767", "1199718703"
    ),
    [string[]]$EvaluationMapSizes = @("16"),
    [string[]]$OpponentNames = @("public_ilialar_risk_averse", "public_arpit_rule_based"),
    [int]$ReplayTimeoutSeconds = 300,
    [int]$MaxReplayAttempts = 1,

    [string]$BaselinePromotion = "outputs\diagnostic_layer\best_agent_ab_counterfactual_v1_same_seeds_16\promotion_metrics.json",
    [string]$BaselineRiskSummary = "outputs\diagnostic_layer\combined_risk_report_best_ab_counterfactual_v1_same_seeds_16\combined_summary.json",
    [string]$SuggestionModel = "outputs\diagnostic_layer\suggestion_reward_lgbm_v1b_from_best\suggestion_reward_lgbm.joblib",
    [string]$LateModel = "outputs\diagnostic_layer\late_big_loss_warning_lgbm_v2_start140_from_best\late_big_loss_warning_lgbm.joblib",

    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipRiskReport,
    [switch]$DryRun,
    [switch]$OverwriteOutput
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Resolve-ProjectPath([string]$PathValue) {
    if ([IO.Path]::IsPathRooted($PathValue)) { return $PathValue }
    return Join-Path $ProjectRoot $PathValue
}

function Invoke-Step([string]$Title, [string[]]$Command) {
    Write-Host ""
    Write-Host "== $Title =="
    Write-Host ($Command -join " ")
    if ($DryRun) { return }
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

$OutputRoot = Resolve-ProjectPath $OutputRoot
$ExperimentDir = Join-Path $OutputRoot $ExperimentName
$AgentDir = Join-Path $ExperimentDir "agent"
$EvalDir = Join-Path $ExperimentDir "eval_same_seeds"
$RiskDir = Join-Path $ExperimentDir "combined_risk_report"
$SummaryDir = Join-Path $ExperimentDir "summary"

New-Item -ItemType Directory -Path $ExperimentDir -Force | Out-Null

$ExperimentConfig = [PSCustomObject]@{
    experiment_name = $ExperimentName
    base_agent = $BaseAgent
    train_script = $TrainScript
    shards_dir = $ShardsDir
    output_agent = $AgentDir
    epochs = $Epochs
    batch_size = $BatchSize
    learning_rate = $LearningRate
    gamma = $Gamma
    max_delta = $MaxDelta
    hidden_channels = $HiddenChannels
    kernel_size = $KernelSize
    p0_residual_scale = $P0ResidualScale
    p1_residual_scale = $P1ResidualScale
    p0_ce_mult = $P0CeMult
    p1_ce_mult = $P1CeMult
    p0_anchor_mult = $P0AnchorMult
    p1_anchor_mult = $P1AnchorMult
    kl_beta = $KlBeta
    l2_beta = $L2Beta
    anchor_weight = $AnchorWeight
    structured_critical_loss = [bool]$StructuredCriticalLoss
    support_worker_mult = $SupportWorkerMult
    support_city_mult = $SupportCityMult
    late_worker_mult = $LateWorkerMult
    late_city_mult = $LateCityMult
    safe_city_mult = $SafeCityMult
    seed = $Seed
    evaluation_seeds = $EvaluationSeeds
    evaluation_map_sizes = $EvaluationMapSizes
    opponents = $OpponentNames
    replay_timeout_seconds = $ReplayTimeoutSeconds
    max_replay_attempts = $MaxReplayAttempts
    created_at = (Get-Date).ToString("o")
}
[IO.File]::WriteAllText(
    (Join-Path $ExperimentDir "experiment_config.json"),
    ($ExperimentConfig | ConvertTo-Json -Depth 8),
    [Text.UTF8Encoding]::new($false)
)

if (-not $SkipTrain) {
    $TrainCommand = @(
        $Python,
        (Resolve-ProjectPath $TrainScript),
        "--shards-dir", (Resolve-ProjectPath $ShardsDir),
        "--agent-dir", (Resolve-ProjectPath $BaseAgent),
        "--output-agent-dir", $AgentDir,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$LearningRate",
        "--gamma", "$Gamma",
        "--max-delta", "$MaxDelta",
        "--hidden-channels", "$HiddenChannels",
        "--kernel-size", "$KernelSize",
        "--p0-residual-scale", "$P0ResidualScale",
        "--p1-residual-scale", "$P1ResidualScale",
        "--p0-ce-mult", "$P0CeMult",
        "--p1-ce-mult", "$P1CeMult",
        "--p0-anchor-mult", "$P0AnchorMult",
        "--p1-anchor-mult", "$P1AnchorMult",
        "--kl-beta", "$KlBeta",
        "--l2-beta", "$L2Beta",
        "--anchor-weight", "$AnchorWeight",
        "--seed", "$Seed"
    )
    if ($StructuredCriticalLoss) {
        $TrainCommand += @(
            "--structured-critical-loss",
            "--support-worker-mult", "$SupportWorkerMult",
            "--support-city-mult", "$SupportCityMult",
            "--late-worker-mult", "$LateWorkerMult",
            "--late-city-mult", "$LateCityMult",
            "--safe-city-mult", "$SafeCityMult"
        )
    }
    if ($OverwriteOutput) { $TrainCommand += "--overwrite-output" }
    Invoke-Step "Train residual agent" $TrainCommand
}

if (-not $SkipEval) {
    $EvalCommand = @(
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "evaluate_agent_generalization.ps1"),
        "-CandidateAgent", $AgentDir,
        "-OutputDir", $EvalDir,
        "-Seeds", ($EvaluationSeeds -join ","),
        "-RandomSeedCount", "0",
        "-EvaluationMapSizes", ($EvaluationMapSizes -join ","),
        "-OpponentNames", ($OpponentNames -join ","),
        "-ReplayTimeoutSeconds", "$ReplayTimeoutSeconds",
        "-MaxReplayAttempts", "$MaxReplayAttempts",
        "-ContinueOnReplayFailure"
    )
    Invoke-Step "Evaluate same-seed games" $EvalCommand
}

if (-not $SkipRiskReport) {
    $ReplayPattern = Join-Path $EvalDir "replays\*.json"
    $RiskCommand = @(
        $Python,
        (Join-Path $PSScriptRoot "score_combined_risk_report.py"),
        $ReplayPattern,
        "--output-dir", $RiskDir,
        "--suggestion-model", (Resolve-ProjectPath $SuggestionModel),
        "--late-model", (Resolve-ProjectPath $LateModel),
        "--map-size", "16",
        "--start-turn", "140",
        "--late-action-threshold", "0.35",
        "--suggestion-action-threshold", "2.0"
    )
    Invoke-Step "Score combined risk report" $RiskCommand
}

$SummaryCommand = @(
    $Python,
    (Join-Path $PSScriptRoot "summarize_experiment.py"),
    "--experiment-name", $ExperimentName,
    "--candidate-agent", $AgentDir,
    "--baseline-agent", (Resolve-ProjectPath $BaseAgent),
    "--candidate-promotion", (Join-Path $EvalDir "promotion_metrics.json"),
    "--baseline-promotion", (Resolve-ProjectPath $BaselinePromotion),
    "--candidate-risk", (Join-Path $RiskDir "combined_summary.json"),
    "--baseline-risk", (Resolve-ProjectPath $BaselineRiskSummary),
    "--output-dir", $SummaryDir
)
Invoke-Step "Summarize experiment" $SummaryCommand

Write-Host ""
Write-Host "Experiment directory: $ExperimentDir"
Write-Host "Agent: $AgentDir"
Write-Host "Summary: $(Join-Path $SummaryDir 'experiment_summary.md')"
