[CmdletBinding()]
param(
    [string]$CandidateAgent = "outputs\auto_league_dagger_v10_shadow\best_agent",
    [string]$OutputDir = "outputs\diagnostic_layer\fresh_seed_gate_dry_run_v1",
    [string[]]$Seeds = @(),
    [int]$RandomSeedCount = 8,
    [string[]]$EvaluationMapSizes = @("16"),
    [string[]]$OpponentNames = @("public_ilialar_risk_averse", "public_arpit_rule_based", "stage400", "first"),
    [int]$ReplayTimeoutSeconds = 300,
    [int]$MaxReplayAttempts = 1,
    [double]$RiskThreshold = 0.35,
    [double]$SafeExpansionThreshold = 0.35,
    [double]$LowRiskThreshold = 0.20,
    [double]$SafetyGateRiskThreshold = 0.70,
    [int]$MaxGateTurn = 160,
    [int]$MinStableRows = 30,
    [double]$MinStableLossRate = 0.60,
    [double]$MinStableBigLossRate = 0.25,
    [switch]$SkipEvaluation,
    [switch]$ContinueOnReplayFailure
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
        "-MaxReplayAttempts", $MaxReplayAttempts
    )
    if ($Seeds.Count -gt 0) {
        $EvalArgs += @("-Seeds", ($Seeds -join ","))
        $EvalArgs += @("-RandomSeedCount", 0)
    }
    if ($ContinueOnReplayFailure) {
        $EvalArgs += "-ContinueOnReplayFailure"
    }
    Write-Host "Running fresh-seed evaluation..."
    & powershell.exe @EvalArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

foreach ($MapSize in $EvaluationMapSizes) {
    Write-Host "Running dry-run diagnostic pipeline for ${MapSize}x${MapSize}..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File (Join-Path $PSScriptRoot "diagnose_evaluation_with_risk_scorer.ps1") `
        -EvaluationDir $OutputDir `
        -MapSize ([int]$MapSize) `
        -RiskThreshold $RiskThreshold `
        -SafeExpansionThreshold $SafeExpansionThreshold `
        -LowRiskThreshold $LowRiskThreshold `
        -SafetyGateRiskThreshold $SafetyGateRiskThreshold `
        -MaxGateTurn $MaxGateTurn
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$SummaryCsv = Join-Path $OutputDir "gate_dry_run\gate_dry_run_summary.csv"
$PolicyJson = Join-Path $OutputDir "intervention_validation\gate_policy_spec.json"
$VerdictPath = Join-Path $OutputDir "dry_run_gate_stability_verdict.json"

$Rows = @()
if (Test-Path -LiteralPath $SummaryCsv) {
    $Rows = @(Import-Csv -LiteralPath $SummaryCsv)
}

$BwRows = @($Rows | Where-Object {
    $_.gate_rule -eq "bw_low_fuel_lt5_high_risk" -and
    ($_.gate_mode -eq "dry_run" -or $_.gate_mode -eq "block")
})
$Stable = $false
$Reason = "bw_low_fuel_lt5_high_risk dry-run rule was not reproduced."
$Metrics = [ordered]@{}
if ($BwRows.Count -gt 0) {
    $Row = $BwRows[0]
    $EventRows = [int][double]$Row.event_rows
    $LossRate = [double]$Row.loss_rate_10
    $BigLossRate = [double]$Row.big_loss_rate_10
    $MeanFutureLoss = [double]$Row.mean_future_loss_10
    $MeanRisk = [double]$Row.mean_risk
    $Stable = (
        $EventRows -ge $MinStableRows -and
        $LossRate -ge $MinStableLossRate -and
        $BigLossRate -ge $MinStableBigLossRate
    )
    $Metrics = [ordered]@{
        gate_mode_seen = $Row.gate_mode
        event_rows = $EventRows
        loss_rate_10 = $LossRate
        big_loss_rate_10 = $BigLossRate
        mean_future_loss_10 = $MeanFutureLoss
        mean_risk = $MeanRisk
    }
    if ($Stable) {
        if ($Row.gate_mode -eq "block") {
            $Reason = "bw_low_fuel_lt5_high_risk reproduced on fresh seeds and validator upgraded it to block; keep runtime usage at log-only/dry-run until broader seeds/opponents confirm."
        } else {
            $Reason = "bw_low_fuel_lt5_high_risk reproduced on fresh seeds; keep a conservative build-worker dry-run gate next."
        }
    } else {
        $Reason = "bw_low_fuel_lt5_high_risk appeared but did not meet stability thresholds; inspect scorer/labels before any gate."
    }
}

$SeedJson = Join-Path $OutputDir "evaluation_seeds.json"
$SeedsRecord = if (Test-Path -LiteralPath $SeedJson) {
    Get-Content -Raw -LiteralPath $SeedJson | ConvertFrom-Json
} else {
    $null
}

$Verdict = [ordered]@{
    stable = $Stable
    reason = $Reason
    candidate_agent = $CandidateAgent
    output_dir = $OutputDir
    evaluation_seeds = $SeedsRecord
    expected_rule = "bw_low_fuel_lt5_high_risk"
    recommended_next_mode = if ($Stable) { "log_only_or_dry_run" } else { "no_gate" }
    thresholds = [ordered]@{
        min_stable_rows = $MinStableRows
        min_stable_loss_rate = $MinStableLossRate
        min_stable_big_loss_rate = $MinStableBigLossRate
        safety_gate_risk_threshold = $SafetyGateRiskThreshold
        max_gate_turn = $MaxGateTurn
    }
    metrics = $Metrics
    files = [ordered]@{
        promotion_metrics = (Join-Path $OutputDir "promotion_metrics.json")
        gate_policy_spec = $PolicyJson
        gate_dry_run_summary = $SummaryCsv
        verdict = $VerdictPath
    }
}

[IO.File]::WriteAllText($VerdictPath, ($Verdict | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
Write-Host "Dry-run gate stability verdict: $VerdictPath"
Write-Host ($Verdict | ConvertTo-Json -Depth 12)
