# P1 Policy Decisions

**Date:** 2026-05-10  
**Status:** ACCEPTED after P0 baseline pass (61/61 tests)  
**Authority:** Source-orchestrator (Hamm)

---

## Optax Compatibility Policy

**Decision:** Keep Optax as optional JAX-extra dependency; no forced core requirement; no rewrite of GSDR/AGSDR now.

**Rationale:**
- Optax is declared in pyproject.toml [jax] extra but has zero uses in current code
- Tests pass without requiring Optax in core paths
- GSDR/AGSDR use custom optimization logic that works correctly
- Future work may add optional thin adapter/wrapper with guarded imports

**Actions:**
- Do NOT rewrite GSDR/AGSDR around Optax during GAMMA
- Keep Optax as optional import in any future adapter
- If adapter is written, guard with try/except imports so core works without Optax
- Document that Optax is available for users who install [jax] extra but not required

---

## Colab Notebook Archival Policy

**Decision:** Archive tfne_izhikevich_net.original.ipynb as Colab artifact; maintain portable tutorials separately.

**Rationale:**
- Notebook contains Colab magic commands (%cd, !pip) and google.colab imports
- These are valid Jupyter syntax but Colab-specific and not portable
- Jupyter AST audit correctly identifies shell magics as non-Python syntax
- Portable tutorials should be separately maintained and nbconvert-runnable

**Actions:**
- Archive tfne_izhikevich_net.original.ipynb with clear `.colab` label or in tutorials/source_notebooks/archive/
- Ensure main tutorials (00-04) are portable Jupyter without magic commands
- Do NOT treat Colab magics as package code errors once artifact is labeled

---

## JAX Parallel APIs Policy (pmap/pjit)

**Decision:** Preserve current CPU-safe device-count fallback behavior; audit before modernization; do NOT aggressively rewrite to pjit yet.

**Rationale:**
- Current code in gsgd.py uses pmap with device-count fallback to vmap
- Pattern is safe and works correctly on CPU-only systems
- Tests pass; device-aware behavior is working
- pjit/sharding modernization is optional future work, not blocking

**Actions:**
- Audit gsgd.py and test_edge_backend.py for current device-aware behavior
- Document device-count fallback pattern in code comments
- Create src/jbiophysic/ops/parallel.py compatibility layer if needed
- Add CPU-only tests that verify fallback behavior
- Modern pjit/NamedSharding migration is separate phase only if tests require it

---

## Izhikevich API Policy

**Decision:** Preserve existing API and green tests; add presets/units/determinism tests later; no premature biological claims.

**Rationale:**
- Existing tests pass (baseline 61/61)
- IzhikevichParams are functionally correct for current use cases
- Parameter presets, unit docs, and determinism tests are enhancements, not fixes
- Truth doctrine: optimizer success is not biological validation; preserve exploratory status

**Actions:**
- Do NOT claim biological correctness until benchmarked against reference data
- Later add RS/FS/chattering presets as optional parameter sets
- Later add unit validation docs (ms, mV, pA/nA)
- Later add deterministic seed tests
- Keep all claims tagged as `exploratory` / `tutorial_exploratory_not_biological_truth`

---

## TFNE Policy (Fields, Sources, Probes)

**Decision:** Later phases add source conservation, gauge, SPD/passivity, and NaN/Inf tests; preserve Emitter→Source→Field→Probe framing; no whole-brain claims.

**Rationale:**
- TFNE is forward-field modeling framework, not whole-brain simulator
- Current tests pass; physics implementation is working
- Advanced validation (conservation, gauge, tensor properties) are robustness enhancements
- Truth doctrine: preserve forward-field framing; do not overclaim scope

**Actions:**
- Later add source conservation tests with tolerance specs
- Later add mean-zero gauge validation tests
- Later add SPD/passivity tensor tests
- Later add NaN/Inf detection tests
- Document TFNE as Emitter→Source→Field→Probe, not whole-brain theory
- Do NOT claim to simulate cortical circuits unless empirically validated

---

## Legacy Cleanup Policy (Deletion by Manifest)

**Decision:** Deletion only via evidence-based manifest; grep/import/reference audits before removal; archive educational material.

**Rationale:**
- scratch/legacy_inspection/ is marked legacy but may contain useful examples
- Some modules (e.g., stability_gatekeeper.py) have no test coverage but may be used
- Deletion without audit trail risks losing recovery option
- Educational/historical code should be archived, not silently deleted

**Actions:**
- Create deletion manifest: for each candidate, document grep findings, test refs, doc refs
- Do NOT delete without full import/reference audit
- Move educational material to scratch/legacy_archive/ with README explaining history
- Archive DynaSim compatibility code if not actively used
- Verify tests remain green after each deletion

---

## Branch Deletion Policy (Final Administrative Phase)

**Decision:** Branch deletion is FINAL ADMINISTRATIVE PHASE ONLY, not part of early GAMMA.

**Rationale:**
- Current non-main branches may contain work in progress or historical reference
- Deletion should happen only after full GAMMA completion and green final tests
- Branch SHAs must be archived with clear manifest
- Explicit human confirmation required before deletion

**Actions:**
- Do NOT delete branches during GAMMA implementation
- After GAMMA final tests pass:
  - Generate branch SHA manifest: `git ls-remote --heads origin | sort > BRANCH_HEADS_ARCHIVE.txt`
  - Create archival tags: `git tag -a archive/branch-name-YYYYMMDD <sha>`
  - Await explicit confirmation: `CONFIRM_DELETE_NON_MAIN_BRANCHES=YES`
  - Then delete: `git push origin --delete branch-name`

Current non-main branches (to be archived later):
- origin/gamma-burst-stabilization
- origin/task/net-eig-izh-jax

---

## Truth Doctrine Alignment

All decisions preserve:
- **truth_safe_unverified:** No biological/scientific claims without empirical validation
- **exploratory_not_production:** Code is research infrastructure, not production system
- **tutorial_not_truth:** Tutorials are executable and educational but not truth claims without benchmarks
- **SI_discipline:** Unit preservation (ms, mV, pA/nA) where applicable
- **TFNE_framing:** Forward-field tool, not whole-brain simulator

---

## Next GAMMA Phase (After Plan Revision)

Narrow, focused implementation targets (NOT all subsystems at once):

1. **Documentation & Plan Alignment** (highest priority)
   - Update README with install modes, truth status, optional dependencies
   - Update pyproject.toml consistency checks
   - Create docs/legacy_cleanup.md

2. **Colab Notebook Archival** (unblock portability)
   - Archive tfne_izhikevich_net.original.ipynb as artifact
   - Verify main tutorials (00-04) are portable
   - Execute main tutorials with nbconvert

3. **JAX Compatibility Audit** (risk mitigation)
   - Scan PRNG patterns in 7 files
   - Document device-count fallback behavior
   - Add CPU-only compatibility tests

Then proceed to remaining GAMMA phases after narrower targets succeed and tests remain green.
