# jbiophysic

`jbiophysic` is an experimental JAX/Jaxley codebase for building simple cortical cell
populations, assembling multi-area hierarchy models, and running biophysical simulations.
It is a research prototype focusing on exploratory modeling rather than a production-ready
simulator.

## Implemented today

- Jaxley-oriented simulation architecture and JAX-native mathematical kernels.
- Biophysical mechanisms: HH-style kinetics, Izhikevich point neurons, and synaptic scaffolds.
- TFNE package namespace under `src/jbiophysic/tfne` for current density, CSD, source
  mollification, gauge fixing, passive tensors, and smoke-testable Poisson-style operators.
- Optimizer scaffolds for SDR, GSDR, AGSDR, and GSGD with explicit bounds/manifests.
- Analysis helpers for spike counts, Fano factor, spectra, synchrony, and LFP/CSD summaries.

## TFNE doctrine

TFNE is treated here as a homogenized, biophysically constrained, bidomain tensor-admittivity
framework for CSD/LFP forward modeling. It is not a validated biological truth model, not a
replacement for Hodgkin-Huxley/Jaxley, and not evidence that a good optimizer fit proves a
biological mechanism.

Implementation must live under `src/jbiophysic/`. The TFNE namespace is therefore
`jbiophysic.tfne`, not a top-level `tfne_jax` package.

## Installation

```bash
pip install -e ".[dev,viz]"
```

For narrow smoke testing without installing optional visualization dependencies:

```bash
PYTHONPATH=src pytest -q tests
```

## Engineering standards

- JAX-compatible kernels are side-effect free.
- State, PRNG keys, and parameter trees are explicit.
- Numerical guards reject NaN/Inf states and preserve source conservation.
- Passive tensors use SPD parameterizations.
- Biological claims remain evidence-calibrated and separate from optimizer success.
