param(
    [string]$CandidateAgent = "outputs\auto_league_dagger_v10_shadow\best_agent",
    [string]$OutputDir = "outputs\diagnostic_layer\fresh_seed_dry_run_gate_v1_safe_16",
    [string[]]$Seeds = @(),
    [int]$RandomSeedCount = 2,
    [string[]]$EvaluationMapSizes = @("16"),
    [string[]]$OpponentNames = @("public_ilialar_risk_averse", "public_arpit_rule_based"),
    [int]$ReplayTimeoutSeconds = 180,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if (-not $SkipEvaluation) {
    $EvalArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "evaluate_agent_generalization.ps1"),
        "-CandidateAgent", $CandidateAgent,
        "-OutputDir", $OutputDir,
        "-RandomSeedCount", $RandomSeedCount,
        "-EvaluationMapSizes", ($EvaluationMapSizes -join ","),
        "-OpponentNames", ($OpponentNames -join ","),
        "-ReplayTimeoutSeconds", $ReplayTimeoutSeconds,
        "-MaxReplayAttempts", "1",
        "-ContinueOnReplayFailure"
    )
    if ($Seeds.Count -gt 0) {
        $EvalArgs += @("-Seeds", ($Seeds -join ","))
    }
    & powershell.exe @EvalArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$LabelCsv = Join-Path $OutputDir "strategy_label_dataset.csv"
$LabelSummary = Join-Path $OutputDir "strategy_label_dataset_summary.json"
$LabelManifest = Join-Path $OutputDir "strategy_label_dataset_manifest.csv"
$CandidateOutput = Join-Path $OutputDir "candidate_action_suggestions"
$GateOutput = Join-Path $OutputDir "dry_run_gate_v1"

& $Python (Join-Path $PSScriptRoot "build_strategy_label_dataset.py") `
    (Join-Path $OutputDir "replays\map_*x*_vs_*_p?.json") `
    --output-csv $LabelCsv `
    --summary-json $LabelSummary `
    --manifest-csv $LabelManifest `
    --map-sizes ($EvaluationMapSizes -join ",")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($EvaluationMapSizes.Count -eq 1 -and [int]$EvaluationMapSizes[0] -eq 16) {
    & $Python (Join-Path $PSScriptRoot "score_candidate_actions_for_best.py") `
        --input-csv $LabelCsv `
        --model-dir (Join-Path $ProjectRoot "outputs\diagnostic_layer\strategy_candidate_scorers_v2_16") `
        --output-dir $CandidateOutput `
        --map-size 16
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Python (Join-Path $PSScriptRoot "dry_run_gate_v1.py") `
        --input (Join-Path $CandidateOutput "candidate_action_suggestions.csv") `
        --output-dir $GateOutput
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Skipping candidate-action dry-run gate scoring: strategy_candidate_scorers_v2_16 is trained for 16x16 only."
    Write-Host "Use the strategy label and promotion outputs for multi-map diagnostics, or train multi-map candidate scorers first."
}

Write-Host "Fresh-seed dry-run gate v1 complete:"
Write-Host (Join-Path $GateOutput "dry_run_gate_summary.json")
