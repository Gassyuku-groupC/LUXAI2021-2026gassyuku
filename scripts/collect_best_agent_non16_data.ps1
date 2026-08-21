param(
    [string]$CandidateAgent = "outputs\auto_league_dagger_v10_shadow\best_agent",
    [string]$OutputRoot = "outputs\diagnostic_layer\best_agent_non16_collection",
    [string[]]$MapSizes = @("12", "24", "32"),
    [string[]]$OpponentNames = @("first", "stage400", "v4_stage350"),
    [int]$RandomSeedCountPerBatch = 2,
    [int]$BatchesPerMap = 1,
    [int]$ReplayTimeoutSeconds = 360
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if (-not [IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

foreach ($MapSize in $MapSizes) {
    foreach ($Batch in 1..$BatchesPerMap) {
        $OutputDir = Join-Path $OutputRoot ("map{0}_batch{1:000}" -f $MapSize, $Batch)
        Write-Host "Collecting map=$MapSize batch=$Batch output=$OutputDir"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass `
            -File (Join-Path $PSScriptRoot "run_fresh_seed_dry_run_gate_v1_safe.ps1") `
            -CandidateAgent $CandidateAgent `
            -OutputDir $OutputDir `
            -RandomSeedCount $RandomSeedCountPerBatch `
            -EvaluationMapSizes $MapSize `
            -OpponentNames ($OpponentNames -join ",") `
            -ReplayTimeoutSeconds $ReplayTimeoutSeconds
        if ($LASTEXITCODE -ne 0) {
            throw "Collection failed for map=$MapSize batch=$Batch"
        }
    }
}

Write-Host "Collection complete: $OutputRoot"
