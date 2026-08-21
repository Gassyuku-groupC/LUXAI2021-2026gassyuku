[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvaluationDir,
    [string]$Model = "outputs\diagnostic_layer\risk_scorer_lgbm_v1_top12_16\risk_scorer_lgbm.joblib",
    [string]$SafeExpansionModel = "outputs\diagnostic_layer\safe_expansion_lgbm_v1c_practical_top12_16\safe_expansion_scorer_lgbm.joblib",
    [int]$MapSize = 16,
    [double]$RiskThreshold = 0.35,
    [double]$SafeExpansionThreshold = 0.35,
    [double]$LowRiskThreshold = 0.20,
    [double]$SafetyGateRiskThreshold = 0.70,
    [int]$ExpansionMinTurn = 80,
    [int]$ExpansionMaxTurn = 0,
    [int]$InterventionTopN = 80,
    [int]$MaxGateTurn = 160,
    [switch]$UseTorchModel
)

$ResolvedEvaluationDir = Resolve-Path $EvaluationDir
$ReplayPattern = Join-Path $ResolvedEvaluationDir "replays\map_${MapSize}x${MapSize}_vs_*_p[01].json"
$FeatureCsv = Join-Path $ResolvedEvaluationDir "strategy_features_for_risk.csv"
$ScoreCsv = Join-Path $ResolvedEvaluationDir "risk_scores.csv"
$SafeScoreCsv = Join-Path $ResolvedEvaluationDir "safe_expansion_scores.csv"
$Report = Join-Path $ResolvedEvaluationDir "risk_report.txt"
$DiagnosticDir = Join-Path $ResolvedEvaluationDir "diagnostic_report"
$InterventionDir = Join-Path $ResolvedEvaluationDir "intervention_candidates"
$ValidationDir = Join-Path $ResolvedEvaluationDir "intervention_validation"
$GateDryRunDir = Join-Path $ResolvedEvaluationDir "gate_dry_run"

Write-Host "Extracting strategy features..."
python scripts\extract_strategy_features.py $ReplayPattern --output $FeatureCsv --map-sizes $MapSize
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Scoring city-loss risk..."
$ScoreScript = if ($UseTorchModel) {
    "scripts\score_city_loss_risk.py"
} else {
    "scripts\score_city_loss_risk_lgbm.py"
}

.\.venv\Scripts\python.exe $ScoreScript `
    --model $Model `
    --input $FeatureCsv `
    --output $ScoreCsv `
    --report $Report `
    --risk-threshold $RiskThreshold
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Scoring safe expansion opportunities..."
.\.venv\Scripts\python.exe scripts\score_safe_expansion_lgbm.py `
    --model $SafeExpansionModel `
    --input $FeatureCsv `
    --output $SafeScoreCsv `
    --report (Join-Path $ResolvedEvaluationDir "safe_expansion_report.csv") `
    --threshold $SafeExpansionThreshold `
    --candidate-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building combined diagnostic report..."
.\.venv\Scripts\python.exe scripts\make_diagnostic_report.py `
    --risk-scores $ScoreCsv `
    --safe-scores $SafeScoreCsv `
    --output-dir $DiagnosticDir `
    --risk-threshold $RiskThreshold `
    --low-risk-threshold $LowRiskThreshold `
    --safe-threshold $SafeExpansionThreshold
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building intervention candidate lists..."
$InterventionArgs = @(
    "scripts\analyze_best_agent_intervention_candidates.py",
    "--risk-scores", $ScoreCsv,
    "--safe-scores", $SafeScoreCsv,
    "--output-dir", $InterventionDir,
    "--safety-risk-threshold", $SafetyGateRiskThreshold,
    "--watch-risk-threshold", $RiskThreshold,
    "--safe-expansion-threshold", $SafeExpansionThreshold,
    "--low-risk-threshold", $LowRiskThreshold,
    "--expansion-min-turn", $ExpansionMinTurn,
    "--top-n", $InterventionTopN
)
if ($ExpansionMaxTurn -gt 0) {
    $InterventionArgs += @("--expansion-max-turn", $ExpansionMaxTurn)
}
.\.venv\Scripts\python.exe @InterventionArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Validating intervention candidates..."
$CombinedCandidates = Join-Path $InterventionDir "intervention_candidates_combined.csv"
$ValidationArgs = @(
    "scripts\validate_intervention_candidates.py",
    "--candidates", $CombinedCandidates,
    "--output-dir", $ValidationDir,
    "--safety-risk-threshold", $SafetyGateRiskThreshold,
    "--max-gate-turn", $MaxGateTurn
)
if ($ExpansionMaxTurn -gt 0) {
    $ValidationArgs += @("--max-expansion-turn", $ExpansionMaxTurn)
}
.\.venv\Scripts\python.exe @ValidationArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Dry-running validated action gate..."
$GatePolicy = Join-Path $ValidationDir "gate_policy_spec.json"
.\.venv\Scripts\python.exe scripts\apply_gate_policy_to_candidates.py `
    --candidates $CombinedCandidates `
    --policy $GatePolicy `
    --output-dir $GateDryRunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Risk report: $Report"
Write-Host "Diagnostic report: $(Join-Path $DiagnosticDir "diagnostic_report.md")"
Write-Host "Intervention candidates: $InterventionDir"
Write-Host "Intervention validation: $ValidationDir"
Write-Host "Gate dry-run: $GateDryRunDir"
