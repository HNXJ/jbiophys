# Spectrolaminar Correct-vs-Inverse Ratio Pairing Audit

Status: evidence-control scaffold, not a simulation result.

Previous bundle status: **PROTOTYPE / NOT CLAIM-ACCEPTED**.

The notebooks were used to extract intended ratio modes and implementation assumptions.
Future runs use repo configs and repo modules, not notebook-embedded duplicated logic.

## Pairing doctrine

Correct and inverse configurations must share seeds, neuron IDs, positions, areas,
layers, total layer counts, TFNE grid/contact geometry, solver/gauge/boundary,
source calibration status, timing, analysis windows, objective weights, and budget.
Only the intended E/PV/SST/VIP layer allocation may differ.

## Parameter table

| parameter | correct notebook/config | inverse notebook/config | paired value | intentionally different? | action |
|---|---|---|---|---:|---|
| truth_mode | truth_safe_unverified | truth_safe_unverified | same | no | required |
| claim_level | controlled_ablation_scaffold | controlled_ablation_scaffold | same | no | required |
| ratio_mode | correct | inverse | differs | yes | intended manipulation |
| seed list | shared list | shared list | same | no | required |
| time_window_ms | [-500,1000] | [-500,1000] | same | no | required |
| event_window_ms | [0,500] | [0,500] | same | no | required |
| post_window_ms | [500,1000] | [500,1000] | same | no | required |
| layer_cell_fractions | deep-high E/I | superficial-high E/I | differs | yes | intended manipulation |
| TFNE field readout | same grid/contact policy | same grid/contact policy | same | no | required |
| solver/gauge/boundary | homogeneous Neumann smoke / mean zero | same | same | no | required |
| source calibration | toy_scale_A_per_native_not_empirical | same | same | no | required |

## Statistical acceptance criteria

Primary metric: paired `Delta S = S_correct - S_inverse`.

Minimum smoke: 2-3 paired seeds, reduced network/time/grid, API and metrics only, no necessity claim.

Minimum serious run: at least 10 paired seeds, paired correct-minus-inverse statistics,
bootstrap or paired permutation CI, mean, median, 95% CI, and per-seed table.

Preferred full run: at least 20 paired seeds if runtime permits, matched random-search/optimization
budget, nulls: layer-shuffle, uniform source, phase-randomized, no-field projection.

Allowed claim after serious run only:

> Under matched TFNE-Izhikevich scaffold conditions, canonical deep-high E/I laminar allocation is required for this model family to reach the declared spectrolaminar readout objective under the tested parameter budget.

Fallback wording:

> Evidence suggests dependence on laminar E/I allocation, but necessity is not established.

## Rejection gates

Reject a run if: stimulus-window spectral peak remains below 15 Hz when gamma is declared target;
gamma increase appears only after per-frequency normalization; raster shows global vertical burst columns;
population firing-rate peaks are implausibly high; synchrony metric indicates collapse; FieldSolution metadata is dropped;
source calibration status is absent; correct/inverse differ by uncontrolled parameters.

## jtfne facade constraint

`src/jbiophysic/jtfne.py` must remain a thin facade/orchestrator. Core logic belongs in
`configs/`, `models/`, `simulation/`, `analysis/`, `optim/`, `viz/`, and `tfne/` modules.
