# Fresh Seed Dry-Run Gate V1 Partial

## Status

The first fresh-seed reproduction run was interrupted because one replay process stalled hard enough to freeze the desktop workflow.

Output directory:

- `outputs/diagnostic_layer/fresh_seed_dry_run_gate_v1b_16`

Evaluation seeds:

- `140725771`
- `87961026`
- `849726517`
- `1741702083`

Completed usable replays:

- 10 replay JSON files
- 7038 strategy-label rows

This is not a full promotion sample, but it is useful as a partial dry-run reproduction.

## Gate Result

Dry-run gate v1 output:

- `outputs/diagnostic_layer/fresh_seed_dry_run_gate_v1b_16/dry_run_gate_v1_partial`

Summary:

- states: 7038
- would_gate: 25
- gate_rate: 0.00355
- gated `bcity`: 23
- gated `bw`: 2
- gated `research`: 0
- gated `no_expand`: 0

Outcome proxy for gated states:

- future 20-turn city-loss rate: 0.52
- mean future team loss in 20 turns: 11.08
- mean final city-tile margin: 42.68

## Interpretation

The fresh-seed partial gate rate is close to the earlier same-seed best-agent dry-run:

- same-seed best gate rate: 0.00373
- fresh partial gate rate: 0.00355

This means the gate is not exploding on fresh seeds. It remains narrow and mostly catches `bcity` in low-buffer/high-risk states.

However, this result should not yet be treated as stable enough for runtime use:

- the sample is partial;
- one public_arpit replay timed out;
- one stage400 replay timed out;
- the run was interrupted during a later public_ilialar replay;
- some opponent/side groups are incomplete, so side-gap checks are not reliable.

## Next Safer Reproduction Plan

Do not run all four opponents and four seeds in one desktop Codex task.

Use smaller batches:

1. public opponents only, 2 seeds, timeout 180 seconds.
2. strong opponents only, 2 seeds, timeout 180 seconds.
3. repeat with another 2 seeds if the first two batches are stable.

Promotion should be read as diagnostic only when both sides are present for each opponent/seed pair.
