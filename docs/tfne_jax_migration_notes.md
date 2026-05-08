# TFNE/JAX migration notes

This package slice migrates only reusable science-library material into the standalone
`jbiophysic` repository. Runtime harnesses, receipts, truth gates, judge/guard logic, and
Gamma orchestration remain outside this repository.

## Safe claims

- TFNE is a forward model for current density, CSD, and LFP-like fields.
- Izhikevich currents are phenomenological and require explicit calibration before being
  interpreted as SI membrane current density.
- GSDR/AGSDR optimizer success is not biological proof.

## First acceptance gates

- compile/import smoke;
- TFNE source conservation;
- SPD/passivity diagnostics;
- finite voltage/current outputs;
- no NaN/Inf;
- deterministic narrow tests.
