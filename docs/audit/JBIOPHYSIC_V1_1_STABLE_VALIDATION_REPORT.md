# jbiophysic V1.1 stable validation report

Status: `ACCEPT_CANDIDATE` for archive/developmental workflow use.

Truth mode: `truth_safe_unverified`.

Claim level: `developmental_demo` / `controlled_ablation_scaffold` for paired configs.

## Scope

This bundle applies second-review readiness edits for the JTFNE/TFNE spectrolaminar workflow:

- deterministic CLI runner
- strict JSON-safe manifests
- operator status export
- motif gate vs profile score split
- physical invariant table export
- cell-type and synchrony diagnostics
- paired correct/inverse ratio configs
- asset hashing
- source-decomposition guardrails
- current-density layout metadata

It does not claim biological proof, calibrated CSD/EEG/MEG amplitude, or E/I-ratio necessity.

## Validation commands run

```bash
PYTHONPATH=src python -m compileall -q src tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_tfne_sources.py tests/test_tfne_tensors_fields.py tests/test_tfne_p0_invariants.py \
  tests/jtfne tests/common tests/objectives tests/analysis tests/configs tests/tfne/test_operator_status.py
PYTHONPATH=src python scripts/run_spectrolaminar_suite.py \
  --config configs/spectrolaminar_v1.yaml \
  --seed 0 \
  --out /mnt/data/jbiophysic_v11_run \
  --smoke
python -m json.tool /mnt/data/jbiophysic_v11_run/manifest.json >/dev/null
```

Observed result:

```text
compile: PASS
targeted tests: 30 passed
CLI smoke: PASS
strict JSON manifest: PASS
```

Full `pytest -q` in this container was attempted but did not complete before timeout after many passing tests. The legacy pipeline test is now opt-in because it exercises old JAX parallel optimization behavior and can dominate runtime. Targeted TFNE/JTFNE/archive-readiness tests passed.

## CLI smoke outputs

The smoke runner writes:

- `manifest.json`
- `metrics.csv`
- `celltype_diagnostics.csv`
- `area_diagnostics.csv`
- `synchrony_diagnostics.csv`
- `field_invariants.csv`
- `operator_status.json`
- `asset_hashes.json`

## Claim boundary

`motif_gate_percent` is an internal gate satisfaction score. `profile_score_percent` is a continuous internal profile score. `S_lam` is reserved for null-normalized similarity and is null unless a declared null distribution is supplied.

The JTFNE source scale is `toy_scale_A_per_native_not_empirical`; physical amplitude claims are refused unless empirical calibration is supplied.
