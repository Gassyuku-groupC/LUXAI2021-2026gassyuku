[CmdletBinding()]
param(
    [string]$RiskScores = "outputs\diagnostic_layer\best_agent_public_opponents_v2_16\risk_scores.csv",
    [string]$SafeExpansionScores = "outputs\diagnostic_layer\best_agent_public_opponents_v2_16\safe_expansion_scores.csv",
    [string]$OutputDir = "outputs\diagnostic_layer\break_reward_loop_v1",
    [double]$SafetyRiskThreshold = 0.70,
    [double]$SafeExpansionThreshold = 0.35,
    [double]$LowRiskThreshold = 0.20,
    [int]$TopN = 500,
    [int]$MaxGateTurn = 160,
    [int]$MaxExpansionTurn = 320,
    [int]$MinGateRows = 8,
    [double]$MinDryRunLossRate = 0.60,
    [double]$MinLossRate = 0.75,
    [double]$MinBigLossRate = 0.35
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath([string]$PathValue) {
    if ([IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $ProjectRoot $PathValue)
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$RiskScoresPath = Resolve-ProjectPath $RiskScores
$SafeExpansionScoresPath = Resolve-ProjectPath $SafeExpansionScores
$OutputRoot = Resolve-ProjectPath $OutputDir
$CandidatesDir = Join-Path $OutputRoot "intervention_candidates"
$ValidationDir = Join-Path $OutputRoot "intervention_validation"
$DryRunDir = Join-Path $OutputRoot "gate_dry_run"
$CandidatesFile = Join-Path $CandidatesDir "intervention_candidates_combined.csv"
$PolicyFile = Join-Path $ValidationDir "gate_policy_spec.json"

foreach ($RequiredPath in @($RiskScoresPath, $SafeExpansionScoresPath)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Missing input file: $RequiredPath"
    }
}

New-Item -ItemType Directory -Path $CandidatesDir, $ValidationDir, $DryRunDir -Force | Out-Null

& $Python (Join-Path $ProjectRoot "scripts\analyze_best_agent_intervention_candidates.py") `
    --risk-scores $RiskScoresPath `
    --safe-scores $SafeExpansionScoresPath `
    --output-dir $CandidatesDir `
    --safety-risk-threshold $SafetyRiskThreshold `
    --safe-expansion-threshold $SafeExpansionThreshold `
    --low-risk-threshold $LowRiskThreshold `
    --top-n $TopN
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $ProjectRoot "scripts\validate_intervention_candidates.py") `
    --candidates $CandidatesFile `
    --output-dir $ValidationDir `
    --safety-risk-threshold $SafetyRiskThreshold `
    --min-gate-rows $MinGateRows `
    --min-dry-run-loss-rate $MinDryRunLossRate `
    --min-loss-rate $MinLossRate `
    --min-big-loss-rate $MinBigLossRate `
    --max-gate-turn $MaxGateTurn `
    --max-expansion-turn $MaxExpansionTurn
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $ProjectRoot "scripts\apply_gate_policy_to_candidates.py") `
    --candidates $CandidatesFile `
    --policy $PolicyFile `
    --output-dir $DryRunDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Intervention validation complete."
Write-Host "Candidates: $CandidatesFile"
Write-Host "Validation report: $(Join-Path $ValidationDir 'intervention_validation_report.md')"
Write-Host "Dry-run summary: $(Join-Path $DryRunDir 'gate_dry_run_summary.csv')"
Write-Host ""
Write-Host "Decision rule:"
Write-Host "- Do not start another reward-weight training run from this output alone."
Write-Host "- Only implement or train a gate after the same approved rules reproduce across fresh seeds."
