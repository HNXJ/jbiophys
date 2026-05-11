# BETA Audit Prompt (jbiophysic)

**Goal:** Read-only assessment of jbiophysic repo state, structure, baseline, and constraints.

**Do NOT edit code, notebooks, or pyproject.toml in BETA phase.**

**Tasks:**

1. **Confirm current checkout state:**
   ```bash
   cd /Users/hamednejat/workspace/main/jbiophysic
   pwd
   git status
   git log --oneline -5
   git branch -r
   ```

2. **Verify Python environment and dependencies:**
   ```bash
   python --version
   pip list | grep -E 'jax|optax|numpy|scipy|pandas'
   ```

3. **Inspect package structure:**
   ```bash
   find src -name '*.py' | head -20
   find tests -name '*.py' | head -20
   ls -la docs/
   ls -la tutorials/
   ```

4. **Run baseline tests (read-only validation):**
   ```bash
   source .venv/bin/activate
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
   # Record: pass/fail count, warnings, execution time
   ```

5. **Audit imports:**
   ```bash
   python -c "import jax; print('JAX:', jax.__version__)"
   python -c "import optax; print('Optax:', optax.__version__)"
   python -c "import jbiophysic; print('jbiophysic imported OK')"
   ```

6. **Inspect critical files (read only):**
   - `README.md` — scope, install, quick validation
   - `pyproject.toml` — dependencies, Python version constraint
   - `AUDIT_AND_REFACTOR_PLAN.md` — prior audit findings (if exists)
   - `docs/` — existing policy docs
   - `src/jbiophysic/__init__.py` — package metadata

7. **Check for issues (no edits):**
   - Syntax errors in core .py files?
   - Missing imports?
   - Orphaned test files?
   - Outdated notebooks?
   - Undocumented policies?

8. **List findings without prescriptions:**
   - Report state as-is
   - Do NOT recommend fixes (that's ALPHA/GAMMA)
   - Do NOT edit any files
   - Do NOT stage or commit

**Output format:**
```
[model][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]

### BETA Audit Summary

**Repo state:**
- Branch: main
- HEAD: [sha]
- origin/main: [sha]
- Clean/dirty: [status]

**Environment:**
- Python: [version]
- JAX: [version], devices: [list]
- Tests: [pass/fail counts]

**Structure findings:**
- [key observations about code layout, docs, notebook state]

**Constraints identified:**
- [Python >= 3.10 required, etc.]

**Prior audit artifacts (if present):**
- [List P0_BASELINE_SUMMARY.md, P1_DECISIONS.md, etc. if found]

**No prescriptions (ALPHA/GAMMA phase job):**
- BETA is assessment only
- Next phase will plan remediation

[model][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]
```

**BETA is complete when:**
- Full inventory of repo state documented
- No edits made
- Findings passed to ALPHA phase
- No assumptions about future direction
