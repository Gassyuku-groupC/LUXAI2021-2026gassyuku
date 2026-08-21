param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$protectedRelative = @(
    "outputs\auto_league_dagger_v10_shadow\best_agent",
    "outputs\auto_league_dagger_v10_shadow\learner_agent",
    "outputs\auto_league_dagger_v7_16x16\best_agent",
    "outputs\auto_league_dagger_v4_16x16\learner_agent"
)

$deleteRelative = @(
    # Large duplicated imitation shards. Keep compact CSV indexes/labels instead.
    "dataset\processed\imitation_shards_expansion_suggestion_v1",
    "dataset\processed\imitation_shards_counterfactual_v4_residual",
    "dataset\processed\imitation_shards_counterfactual_v4",
    "dataset\processed\imitation_shards_counterfactual_v3",
    "dataset\processed\imitation_shards_counterfactual_v2",
    "dataset\processed\imitation_shards_counterfactual_v1",
    "dataset\processed\auxiliary_labels_counterfactual_v4_residual",
    "dataset\processed\auxiliary_labels_counterfactual_v4_residual_smoke",

    # Retired output branches.
    "outputs\auto_league_16x16",
    "outputs\auto_league_dagger_v2_16x16",
    "outputs\auto_league_dagger_v3_16x16",
    "outputs\auto_league_dagger_v5_16x16",
    "outputs\auto_league_dagger_v6_16x16",
    "outputs\auto_league_dagger_v7_refine_16x16",
    "outputs\auto_league_dagger_v8_generalization",
    "outputs\auto_league_dagger_v9_robustness",
    "outputs\auto_league_rl_adapt_v13_from_best",
    "outputs\auto_league_scale_loss_v14",
    "outputs\auto_league_scale_loss_v14b_from_stage48",
    "outputs\auto_league_safe_scale_v14c_from_stage48",
    "outputs\auto_league_aux_risk_v15_from_best",
    "outputs\auto_league_aux_risk_v15b_from_best",
    "outputs\auto_league_survival_gate_v16_from_best",
    "outputs\auto_league_critical_survival_v17_from_best",
    "outputs\auto_league_local_survival_v18_from_best",
    "outputs\auto_league_balanced_local_survival_v19_from_best",
    "outputs\counterfactual_bc_v1_from_best",
    "outputs\counterfactual_bc_v2_from_best",
    "outputs\counterfactual_bc_v3_from_best",
    "outputs\counterfactual_bc_v4_from_best",
    "outputs\expansion_suggestion_bc_v1_from_best",
    "outputs\residual_head_v1_from_best",
    "outputs\residual_head_v1b_from_best",
    "outputs\spatial_residual_head_v2_smoke_from_best",
    "outputs\spatial_residual_head_v2c_smoke_from_best",
    "outputs\spatial_residual_head_v2d_smoke_from_best",
    "outputs\spatial_residual_head_v2e_smoke_from_best",
    "outputs\auxiliary_risk_head_smoke",
    "outputs\auxiliary_risk_head_v1",
    "outputs\auxiliary_risk_head_v2",
    "outputs\auxiliary_risk_head_v2_smoke",
    "outputs\fuel_support_v1_from_best_agent",
    "outputs\fuel_support_v2_from_best_agent",
    "outputs\fuel_support_v3_from_best_agent",
    "outputs\risk_rules_baseline_best",
    "outputs\risk_rules_v1_from_best",
    "outputs\risk_rules_v2_from_best",
    "outputs\risk_rules_v3_from_best",
    "outputs\risk_rules_v4_from_best",
    "outputs\risk_rules_v5_from_best",
    "outputs\risk_rules_v6_from_best",
    "outputs\risk_rules_v7_from_best",
    "outputs\risk_rules_v7b_from_best",
    "outputs\survival_research_finetune_16x16",
    "outputs\survival_research_buffer2_finetune_16x16",
    "outputs\teacher_finetune_16x16",
    "outputs\experiments",
    "outputs\audits",
    "outputs\diagnostics",
    "outputs\risk_feature_logs",

    # Old diagnostic output folders. Keep only active diagnostic route folders.
    "outputs\diagnostic_layer\expansion_suggestion_bc_v1_eval_same_seeds_16",
    "outputs\diagnostic_layer\expansion_suggestion_bc_v1_dry_run_gate_v1_16",
    "outputs\diagnostic_layer\fresh_seed_gate_dry_run_v1",
    "outputs\diagnostic_layer\counterfactual_bc_v1_eval_v3_heldout_16",
    "outputs\diagnostic_layer\counterfactual_bc_v2_eval_same_seeds_16",
    "outputs\diagnostic_layer\counterfactual_bc_v3_eval_same_seeds_16",
    "outputs\diagnostic_layer\counterfactual_bc_v4_eval_same_seeds_16",
    "outputs\diagnostic_layer\best_agent_ab_counterfactual_v1_same_seeds_16",
    "outputs\diagnostic_layer\residual_head_v1b_eval_same_seeds_16",
    "outputs\diagnostic_layer\best_agent_public_opponents_v3_heldout_16",
    "outputs\diagnostic_layer\best_agent_public_opponents_v2_16",
    "outputs\diagnostic_layer\best_agent_strong_first_same_seeds_v2c_random8",
    "outputs\diagnostic_layer\best_agent_public_opponents_v1_16",
    "outputs\diagnostic_layer\best_agent_v10_random2_16_allopponents",
    "outputs\diagnostic_layer\fuel_support_v2_seed1259068876_16_fresh",
    "outputs\diagnostic_layer\best_agent_seed1259068876_16_baseline"
)

$pruneChildren = @(
    "outputs\auto_league_dagger_v7_16x16\game_stage_*",
    "outputs\auto_league_dagger_v4_16x16\game_stage_*"
)

function Resolve-ProjectPath([string]$RelativePath) {
    $candidate = Join-Path $ProjectRoot $RelativePath
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    return $null
}

function Test-IsInsideProject([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    return $full.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)
}

function Test-IsProtected([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    foreach ($rel in $protectedRelative) {
        $protected = Join-Path $ProjectRoot $rel
        if (Test-Path -LiteralPath $protected) {
            $protectedFull = (Resolve-Path -LiteralPath $protected).Path
            if ($full.Equals($protectedFull, [StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
            if ($full.StartsWith($protectedFull + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
    }
    return $false
}

$targets = New-Object System.Collections.Generic.List[string]
foreach ($rel in $deleteRelative) {
    $resolved = Resolve-ProjectPath $rel
    if ($resolved) {
        $targets.Add($resolved)
    }
}
foreach ($pattern in $pruneChildren) {
    Get-ChildItem -Path (Join-Path $ProjectRoot $pattern) -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $targets.Add($_.FullName)
    }
}

$uniqueTargets = $targets | Sort-Object -Unique
$rows = @()
foreach ($target in $uniqueTargets) {
    if (-not (Test-IsInsideProject $target)) {
        throw "Refusing to delete outside project: $target"
    }
    if (Test-IsProtected $target) {
        throw "Refusing to delete protected path: $target"
    }
    $sum = (Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $rows += [pscustomobject]@{
        Path = $target
        SizeGB = [math]::Round(($sum / 1GB), 3)
    }
}

$total = [math]::Round((($rows | Measure-Object SizeGB -Sum).Sum), 3)
Write-Host "Project root: $ProjectRoot"
Write-Host "Mode: $(if ($Execute) { 'EXECUTE' } else { 'DRY RUN' })"
Write-Host "Targets: $($rows.Count)"
Write-Host "Estimated reclaim: $total GB"
$rows | Sort-Object SizeGB -Descending | Format-Table -AutoSize

if (-not $Execute) {
    Write-Host "Dry run only. Re-run with -Execute to delete these targets."
    exit 0
}

foreach ($row in $rows) {
    Remove-Item -LiteralPath $row.Path -Recurse -Force
}
Write-Host "Cleanup complete."
