# jbiophysic Audit and Refactor Plan

**Date:** 2026-05-10  
**Root:** /Users/hamednejat/workspace/main/jbiophysic  
**Branch:** main  
**Initial HEAD:** 1ad502d073451d5b13db7264deb1155d9166d0a6  
**Current HEAD:** 1ad502d073451d5b13db7264deb1155d9166d0a6  
**Status:** BETA/ALPHA complete, P0 baseline PASS, P1 plan revised  

## P0 Baseline Status (2026-05-10, 10:45)

✅ **BASELINE PASS** — Python 3.11 environment, all tests green.

- **Python environment:** 3.11.15 in `.venv/`, editable install succeeded
- **Test baseline:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q` → **61 passed, 0 failed, 0 errors, 2 warnings (non-critical)**
- **JAX import:** jax==0.10.0, devices=[CpuDevice(id=0)]
- **Optax import:** optax==0.2.8 (successfully imported and available)
- **Core module imports:** All 7 focused modules import OK (jbiophysic, cells.izhikevich, cells.hh, tfne, tfne.sources, tfne.fields, tfne.solvers)
- **Installation command:** `python -m pip install -e '.[dev,jax,tutorials]'` — SUCCESS

This establishes that the codebase is runnable, testable, and ready for controlled refactoring.

## BETA Audit Findings

### Package Status
- **Syntax:** ✅ Clean - 154 Python files, no syntax errors
- **Dependencies:** ✅ Resolved - Python >=3.10 satisfied by 3.11 environment
  - 13 Jupyter notebooks (5 main, 5 executed, 3 source)
- **JAX Usage:** 53 imports across 30+ modules
- **JAX NumPy:** 52 uses across 30+ modules
- **Optax:** 0 uses in core code; declared optional in [jax] extra; import available
- **pmap/pjit:** 2 files use (gsgd.py with device-count fallback; tests/test_edge_backend.py) — currently safe but pre-modern
- **PRNG Usage:** 7 files (audit for explicit key discipline)

### Notebook Status
- Main tutorials: ✅ AST clean, no newline collapse, sequential exec counts
- Source notebooks: ⚠️ tfne_izhikevich_net.original.ipynb contains Colab magic (`%cd`, `!pip`) and google.colab imports — valid Jupyter but Colab-specific artifact
- Policy: Archive/label as Colab artifact; maintain portable tutorials separately
- Missing features: Optax integration tutorials (optional, not core)

### Module Inventory
- **cells:** izhikevich.py, hh.py (core neuron models)
- **tfne:** fields, sources, tensors, physics (forward modeling)
- **networks:** connectivity, hierarchy builders
- **models:** builders (rate_models, reduced_models), optimization (agsdr, gsgd), simulation
- **optim:** gsdr.py, gsgd.py, sdr.py, bounds.py
- **ops:** firing.py, lfp.py, stats.py (analysis)
- **scratch:** legacy_inspection, verify_fixes.py (cleanup candidates)

### Suspected Legacy/Stale Modules
- `scratch/legacy_inspection/*` (marked as legacy, candidate for archival)
- `models/training/stability_gatekeeper.py` (no test coverage found)
- Old DynaSim compatibility code (dynasim_smoke_test.py in scratch)

### Resolved & Policy Decisions
1. ✅ **Python version:** 3.11 environment created; tests pass
2. **Optax:** Keep as optional JAX-extra compatibility; no forced core dependency; no rewrite of GSDR/AGSDR now; optional adapter later with guarded imports
3. **pmap/pjit:** Preserve current CPU-safe behavior; audit before modernization; add tests documenting current device-count fallback
4. **Colab notebook:** Archive tfne_izhikevich_net.original.ipynb as historical artifact; portable tutorials separate
5. **Audit scripts:** audit_repo.py hardcoded path is in script only, not package code — ignore

## GAMMA Phase Priorities (Revised with P1 Policies)

### A. Packaging and Formatting
- [x] ✅ Python requirement validated at >=3.10 (3.11 environment confirmed)
- [ ] Validate pyproject.toml syntax (check for newline collapses)
- [ ] Verify version consistency between pyproject.toml and src/jbiophysic/__init__.py
- [ ] Dependency structure already well-organized; no refactoring needed

### B. JAX Compatibility Audit
- [ ] Scan for: Python mutation in jit, implicit numpy/JAX mixing, non-static shapes
- [ ] Establish deterministic PRNG discipline (7 files to audit)
- [ ] Ensure pytrees for outputs
- [ ] Check for over-jitting

### C. Optax Compatibility (Optional Adapter)
- **Policy decision:** Keep Optax as optional JAX-extra; do not force core dependency; do not rewrite GSDR/AGSDR around Optax yet
- [ ] Document that Optax is available but not required for core functionality
- [ ] Later: Create optional thin adapter/wrapper for SGD/Adam/AdamW interop with guarded imports
- [ ] Later: Add optional integration tests if adapter is written
- [ ] Keep core GSDR/AGSDR unchanged; Optax rewrite is future work only if needed

### D. pmap/pjit Audit & Compatibility Layer (NOT aggressive rewrite)
- **Policy decision:** Preserve current CPU-safe device-count fallback behavior; do not aggressively rewrite to pjit yet
- [ ] Audit gsgd.py pmap usage: document current fallback-to-vmap behavior
- [ ] Audit tests/test_edge_backend.py pjit usage
- [ ] Create src/jbiophysic/ops/parallel.py compatibility layer documenting device-count aware behavior
- [ ] Add tests that pass on CPU-only and document current limitations
- [ ] Modern pjit/sharding migration is future phase, not now

### E. Izhikevich API Robustness
- [ ] Verify IzhikevichParams correctness (tests already pass)
- [ ] Later: Add parameter presets (RS, FS, chattering)
- [ ] Later: Add unit validation docs (ms, mV, pA/nA)
- [ ] Later: Add deterministic seed tests
- [ ] Ensure JIT compatibility (tests pass; assume safe for now)

### F. TFNE Validation (Later Phase)
- [ ] Verify Emitter -> Source -> Field -> Probe model (tests pass; preserve framing)
- [ ] Later: Add source conservation tests with tolerance spec
- [ ] Later: Add mean-zero gauge tests
- [ ] Later: Add SPD/passivity tensor tests
- [ ] Later: Add NaN/Inf detection tests
- [ ] Do not claim biological validation; exploratory framework only

### G. Tutorials and Docs
- [ ] Archive tfne_izhikevich_net.original.ipynb as Colab artifact (has %cd, !pip, google.colab)
- [ ] Ensure main tutorials (00-04) are portable Jupyter (nbconvert-runnable)
- [ ] Execute main tutorials (00-04) with nbconvert
- [ ] Update README with install modes, truth status, optional dependency notes
- [ ] Create docs/izhikevich_api.md
- [ ] Create docs/legacy_cleanup.md (for archived/deleted modules)
- [ ] Optional: docs/optax_quickstart.md only if adapter is written

### H. Legacy Code Removal (By Manifest Only)
- **Policy decision:** Deletion only by evidence-based manifest; grep/import/reference checks before removal
- [ ] Identify deletion candidates: scratch/legacy_inspection/*, verify_fixes.py, dynasim_smoke_test.py
- [ ] For each candidate: check if it is imported, referenced in tests/docs/README, or needed for examples
- [ ] Move educational/historical material to scratch/legacy_archive/ with manifest
- [ ] Delete only unused code with full grep audit trail
- [ ] Verify tests remain green after deletion

### I. CI/Test Hardening
- [x] ✅ Tests run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q (61/61 pass)
- [ ] Add @pytest.mark.fast to smoke tests for CI speed
- [ ] Add determinism test: same seed → same result
- [ ] Ensure bounded test execution time

## Branch Retirement Policy

**FINAL ADMINISTRATIVE PHASE ONLY** — not part of early GAMMA.

- Do not delete non-main branches during GAMMA implementation
- Only after full tests pass and explicit confirmation received:
  - Generate branch SHA manifest (git ls-remote --heads origin | sort)
  - Create archival tags for non-main branches
  - Await explicit `CONFIRM_DELETE_NON_MAIN_BRANCHES=YES` confirmation
  - Then delete non-main branches via git push origin --delete

Current non-main branches:
- origin/gamma-burst-stabilization
- origin/task/net-eig-izh-jax

## Stop Conditions (MANDATORY)
- ❌ Do not delete code without evidence-based manifest
- ❌ Do not use `git add .`
- ❌ Do not delete branches without explicit final confirmation gate
- ✅ P0 baseline pass allows GAMMA implementation to proceed

## Next Action
**P1 PLAN REVISION IN PROGRESS:** Update AUDIT_AND_REFACTOR_PLAN.md to incorporate P0 evidence and P1 policy decisions. Create P1_DECISIONS.md and P0_BASELINE_SUMMARY.md. After revision complete, proceed to narrow GAMMA phase: documentation/plan alignment + Colab notebook archival decision.

