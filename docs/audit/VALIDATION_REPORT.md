# jbiophysic TFNE/JAX science-slice validation report

Date: 2026-05-08
Truth mode: `truth_safe_unverified`
Scope: standalone `HNXJ/jbiophysic` science-library migration slice under `src/jbiophysic/`.

## Implemented slice

- `jbiophysic.tfne`: regular grids, gauge fixing, source mollifiers, source conservation, SPD tensors, finite-difference current/CSD helpers, smoke Jacobi Poisson helper, validation guards.
- `jbiophysic.cells`: Izhikevich and HH smoke primitives.
- `jbiophysic.models`: TFNE-Izhikevich explicit calibration bridge and omission-lite condition labels.
- `jbiophysic.optim`: SDR/GSDR/AGSDR/GSGD helpers, bounds, manifests.
- `jbiophysic.analysis`: spikes, Fano factor, spectra, synchrony, LFP/CSD summaries.
- `jbiophysic.networks`, `synapses`, `pipelines`, `viz`, `io`: minimal reusable scaffolds.

## Commands run in sandbox

```bash
cd /mnt/data/jbiophysic_work
python -m compileall -q src tests
PYTHONPATH=src python - <<'PY'
import jax.numpy as jnp
from jbiophysic.tfne import make_regular_grid, gaussian_mollifier, conservation_error
from jbiophysic.pipelines.simulate import run_izhikevich_constant_current

grid = make_regular_grid((5, 5, 5), (20e-6, 20e-6, 20e-6))
eta = gaussian_mollifier(grid, jnp.array([40e-6, 40e-6, 40e-6]), 25e-6)
err = conservation_error(grid, eta * 1e-12, 1e-12)
smoke = run_izhikevich_constant_current(T_ms=50.0, dt_ms=0.5, I=10.0)
print('TFNE_SOURCE_ERR_A', float(err))
print('IZH_N_SPIKES', smoke['n_spikes'])
assert abs(float(err)) < 1e-18
assert smoke['n_spikes'] >= 1
PY
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
```

## Observed output

```text
TFNE_SOURCE_ERR_A 0.0
IZH_N_SPIKES 2
...............                                                          [100%]
15 passed in 19.69s
```

## Important caveat

The sandbox `git clone https://github.com/HNXJ/jbiophysic.git` attempt failed because the sandbox container could not resolve `github.com`. The repo identity and current public layout were verified through GitHub web access, and this bundle is repo-shaped for application by a Mac CLI worker with normal GitHub access.
