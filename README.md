# jbiophysic

Experimental computational neuroscience framework for:

- Izhikevich and HH-style neuron models
- laminar E/PV/SST/VIP cortical circuits
- multi-area low/mid/high cortical hierarchy simulations
- global oddball and omission task scaffolds
- TFNE forward-field CSD/LFP modeling
- optimization and plasticity experiments

## Status and Requirements

**Scope:** Exploratory research infrastructure for computational neuroscience. Not a validated biological simulator. Optimizer success is not biological proof.

**Python:** Requires Python >=3.10. Baseline validated on Python 3.11.15 with 61/61 tests passing.

**Dependencies:**
- **Core:** numpy, scipy, pandas, PyYAML (minimal)
- **JAX:** jax, jaxlib, equinox, optax, diffrax (optional, [jax] extra)
- **Tutorials:** jupyter, nbformat, nbconvert, ipykernel, matplotlib (optional, [tutorials] extra)
- **Development:** pytest, pytest-cov, ruff, black (optional, [dev] extra)

**JAX & Optax Status:**
- JAX (0.10.0): Core package uses 53 imports, 52 jax.numpy uses. CPU-safe baseline.
- Optax (0.2.8): Available as optional [jax] extra; not required by core imports.
- pmap/pjit: Current fallback-to-vmap CPU behavior is preserved; modernization is optional future work.
- PRNG: Explicit key discipline enforced; same seed → same result.

## Install

Minimal (core only):

```bash
pip install -e .
```

Development (tests):

```bash
pip install -e ".[dev]"
```

JAX stack (neural modeling):

```bash
pip install -e ".[jax]"
```

Tutorials (executable notebooks):

```bash
pip install -e ".[tutorials]"
```

Full stack (everything):

```bash
pip install -e ".[jax,tutorials,dev]"
```

## Quick Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q
```

Expected: 61 passed, 0 failed.

## Tutorials

**Portable tutorials (nbconvert-executable, no magic commands):**

- `tutorials/00_neuronal_equations_book.ipynb` — Equation families overview
- `tutorials/01_izhikevich_hh_single_neurons.ipynb` — Izhikevich and HH single neurons
- `tutorials/02_tfne_forward_fields.ipynb` — TFNE forward-field modeling
- `tutorials/03_tfne_izhikevich_hybrid.ipynb` — Izhikevich-to-TFNE hybrid network
- `tutorials/04_laminar_oddball_three_area_cortex.ipynb` — Laminar cortex scaffold

These are executable teaching artifacts and should not be treated as validated biological claims. See `tutorials/README.md` for scientific guardrails and replication constraints.

**Colab artifacts (historical reference, not portable):**

- `tutorials/source_notebooks/tfne_izhikevich_net.colab.ipynb` — Original Colab notebook with google.colab imports and shell magics (%cd, !pip). For reference only; use portable tutorials for executable work.

HTML exports of portable tutorials live in `tutorials/html/`.
