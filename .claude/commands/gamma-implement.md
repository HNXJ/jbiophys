# GAMMA Implementation Prompt (jbiophysic)

**Goal:** Execute one narrow GAMMA phase with validation and reporting.

**Before starting:**
```bash
cd /Users/hamednejat/workspace/main/jbiophysic
git status --short --branch
git fetch origin
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed
```

**Scope constraints:**
- One GAMMA phase only (no parallelization)
- No source refactoring beyond stated phase scope
- No Optax integration (kept optional)
- No pmap/pjit rewriting (preserve CPU-safe behavior)
- No branch deletion (final phase only)
- No broad `git add .`

**Workflow:**
1. Confirm baseline (Section "Before starting")
2. Review CLAUDE.md Section relevant to phase (F for JAX, G for Optax, etc.)
3. Read approved phase in AUDIT_AND_REFACTOR_PLAN.md or P1_DECISIONS.md
4. Make changes in narrow scope
5. Run validation: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short`
6. Stage exact files: `git add <path1> <path2>`
7. Commit: `git commit -m "category: message"`
8. Fetch/rebase/push: `git fetch origin && git pull --rebase origin main && git push origin main`
9. End with identity-wrapped report (CLAUDE.md Section M)

**Approved GAMMA phases:**
- **GAMMA 1** (done): README/docs alignment, Colab artifact archival
- **GAMMA 2** (next): JAX compatibility audit with tests
- **GAMMA 3**: Optax compatibility (decide on optional adapter)
- **GAMMA 4**: Izhikevich API hardening (presets, docs, determinism)
- **GAMMA 5**: TFNE validation (conservation, gauge, SPD)
- **GAMMA 6**: pmap/pjit audit with compatibility layer
- **GAMMA 7**: Legacy code cleanup (deletion manifest)
- **GAMMA 8**: Branch retirement (manifest, tags, deletion)

**If tests fail or conflicts arise:**
- Do NOT force push
- Diagnose locally
- Report with risks (CLAUDE.md Section M)
- Do NOT proceed until resolved

**Truth status:** truth_safe_unverified (unless phase produces validated receipts)
