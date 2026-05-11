# THETA Validation Prompt (jbiophysic)

**Goal:** Validate all GAMMA phases complete, tests green, and produce final report.

**Only run after all GAMMA phases are pushed and verified on origin/main.**

**Pre-validation checklist:**
```bash
cd /Users/hamednejat/workspace/main/jbiophysic
git status --short --branch
# Expected: clean working directory, main = origin/main

git fetch origin
git rev-parse origin/main  # Note SHA

source .venv/bin/activate
python --version  # Should be 3.11.15

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short
# Expected: 61 passed, 0 failed (same baseline as P0)
```

**Validation tasks:**
1. No uncommitted changes
2. All GAMMA phases committed and pushed
3. 61/61 tests passing
4. No secrets in any context files: `grep -RInE 'api[_-]?key|secret|token|password|private[_-]?key|bearer|BEGIN .*PRIVATE KEY' src/ tests/ docs/ CLAUDE.md .claude 2>/dev/null || true`
5. All README, docs, and CLAUDE.md files accurate to current state
6. No deprecated or broken file references in docs

**Final report format (CLAUDE.md Section M):**

```
[claude-haiku-4-5-20251001][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]

### Final Validation Report

**GAMMA Phases Completed:**
- GAMMA 1: README/docs alignment (commit aaa83cb)
- GAMMA 2: JAX compatibility audit (commit XXX)
- [...]

**Test Baseline:**
- Before GAMMA: 61 passed, 0 failed
- After GAMMA: 61 passed, 0 failed
- Regression check: PASS

**Documentation:**
- CLAUDE.md: operational context finalized
- README.md: install modes and baseline evidence aligned
- docs/jax_compatibility.md: JAX policy documented
- docs/tutorial_status.md: notebook classification complete
- P0_BASELINE_SUMMARY.md: preserved as historical record
- P1_DECISIONS.md: policy decisions documented

**No-secrets scan:** PASS (no actual secrets found)

**Staging and commit history:**
- All files staged via exact paths (no `git add .`)
- All commits follow message convention
- All pushes to origin/main successful

### Truth Status
- truth_mode: truth_safe_unverified
- truth_bearing_run: false (no new validated scientific evidence)
- Claim: Repository is exploration infrastructure with validated tests, not biological truth.

### Decision
- ACCEPT_CANDIDATE: All GAMMA phases complete, tests green, context finalized
- REVISE: [if issues found]
- REJECT_INVALID: [if critical blocker]

[claude-haiku-4-5-20251001][/Users/hamednejat/workspace/main/jbiophysic][yyyymmdd-hhmm]
```

**If any issue found:**
- Do NOT declare success
- Document issue with specific file/line/evidence
- Propose remediation
- Create follow-up GAMMA phase if needed
- Report constraints honored or violated

**No force push, no deletions without separate approval.**
