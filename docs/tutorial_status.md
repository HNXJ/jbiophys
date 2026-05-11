# Tutorial Status and Classification

**Date:** 2026-05-10  
**Status:** P0 baseline validated, artifacts classified

---

## Tutorial Classification

Notebooks in this project are classified into three categories:

### 1. Portable Tutorials (executable, reproducible)

**Properties:**
- No Jupyter magic commands (%cd, !shell)
- No google.colab or environment-specific imports
- Pure Python + JAX/NumPy
- nbconvert-executable without modification
- Runnable on any environment with dependencies installed
- Teaching artifacts; not biological validation

**Current portable tutorials:**
- `tutorials/00_neuronal_equations_book.ipynb` — Equation families overview
- `tutorials/01_izhikevich_hh_single_neurons.ipynb` — Single-neuron models
- `tutorials/02_tfne_forward_fields.ipynb` — Forward-field modeling
- `tutorials/03_tfne_izhikevich_hybrid.ipynb` — Izhikevich-TFNE bridge
- `tutorials/04_laminar_oddball_three_area_cortex.ipynb` — Laminar cortex scaffold

**Validation status:**
- AST parsing: ✅ Clean, no syntax errors
- Execution counts: ✅ Sequential, present
- Outputs: ✅ Present and up-to-date
- Test coverage: ✅ Covered by test_import_smoke.py

**How to execute:**
```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/00_neuronal_equations_book.ipynb --inplace
```

---

### 2. Colab Artifacts (historical reference, environment-specific)

**Properties:**
- Contains `google.colab` imports
- Contains Jupyter magic commands (%cd, %magics, !shell)
- Colab-specific paths and environment setup
- Not portable to standard Jupyter or Python environments
- Preserved for historical/reference purposes
- NOT executable in standard environments without modification

**Current Colab artifacts:**
- `tutorials/source_notebooks/tfne_izhikevich_net.colab.ipynb` — Original Colab development notebook
  - Evidence: `google.colab.drive`, `%cd`, `!pip install`
  - Date: Preserved from original development
  - Status: Reference only; use portable tutorials for teaching/reproduction

**What to do with Colab artifacts:**
- Keep for reference and historical context
- Do NOT execute in standard Jupyter
- Do NOT include in AST syntax validation
- Do NOT run nbconvert execution on these
- Label clearly in documentation

---

### 3. Source/Reference Notebooks (works-in-progress, development)

**Properties:**
- Intermediate development notebooks
- May contain scratch cells or exploration
- Used to derive portable tutorials
- Not guaranteed to execute without modification
- Useful for understanding tutorial derivation

**Current source notebooks:**
- `tutorials/source_notebooks/neural_simulations_tutorial.ipynb` — Working version of neural simulations
- `tutorials/source_notebooks/neural_simulations_tutorial.original.ipynb` — Original source

**Status:**
- AST parsing: 0 errors but execution counts may be incomplete
- Use: Reference only; prefer portable tutorials for teaching

---

## Notebook Audit Evidence

From `AUDIT_NOTEBOOKS_INITIAL.txt`:

| Notebook | Portable? | AST errors | Outputs | Status |
|----------|-----------|-----------|---------|--------|
| 00_neuronal_equations_book.ipynb | ✅ | 0 | ✅ | Ready |
| 01_izhikevich_hh_single_neurons.ipynb | ✅ | 0 | ✅ | Ready |
| 02_tfne_forward_fields.ipynb | ✅ | 0 | ✅ | Ready |
| 03_tfne_izhikevich_hybrid.ipynb | ✅ | 0 | ✅ | Ready |
| 04_laminar_oddball_three_area_cortex.ipynb | ✅ | 0 | ✅ | Ready |
| tfne_izhikevich_net.colab.ipynb | ❌ Colab | 2 (magics) | ✅ | Reference |
| neural_simulations_tutorial.ipynb | ⚠️ Source | 0 | none | Reference |

---

## Running Tutorials

**To execute portable tutorials:**

From repo root with Python >=3.10 + dev extras:

```bash
pip install -e ".[dev,tutorials]"
PYTHONPATH=src jupyter nbconvert --to notebook --execute tutorials/00_neuronal_equations_book.ipynb --inplace
```

**Expected results:**
- Execution without errors
- Cell execution counts match original
- Outputs generated and stored
- HTML versions can be exported to `tutorials/html/`

---

## Scientific Guardrails

All tutorials carry the following caveats:

1. **Exploratory, not biological truth:**
   - Models are executable examples of neural equations.
   - Parameters are not fitted to experimental data.
   - No claim of biological realism without empirical validation.

2. **Izhikevich scope:**
   - Native current-like drive, not nanoamperes.
   - Unit validation required for publication.
   - No membrane biophysics beyond spiking equation.

3. **TFNE scope:**
   - Forward-field tool for CSD/LFP modeling.
   - Not a whole-brain simulator.
   - Calibration required before source-to-field projection.

4. **Task scaffolds:**
   - Oddball and omission timings are illustrative.
   - No claim of replicating published findings without separate validation.

See `tutorials/README.md` for detailed guardrails and `tutorials/data/` for replication manifests.

---

## Next Steps

- **P0:** Portable tutorials validated and listed (done)
- **P1:** Colab artifacts labeled and archived (done)
- **P2 (future):** Execute all portable tutorials as part of CI/CD
- **P3 (future):** Add tutorial validation to test suite
