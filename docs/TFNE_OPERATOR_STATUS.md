# TFNE Operator Status

This document is synchronized with `src/jbiophysic/tfne/operator_status.py`.
It prevents manuscript/repo claim inflation by distinguishing implemented modules,
prototypes, partial modules, and specified future modules.

The formal extended TFNE stack is:

```text
E_theta -> S_WDR -> C_mu_nu -> Q_eta_alpha -> F_Omega_B_G_Gamma -> P -> A -> O -> C
```

Current V1.1 status:

| Operator | Status | Claim allowed | Claim forbidden |
|---|---|---|---|
| E_theta emitter | repo_module | emitters generate timing/native/source-proxy states | reduced native current is not amperes without calibration |
| S_WDR synapse | partial_repo_module | directed effective-weight scaffolds | complete receptor tensor claim |
| C_mu_nu chemical | specified_future_module | future parameter modulator | q_chem baseline source-density claim |
| Q_eta_alpha source projection | partial_repo_module | normalized/source-sink proxy checks | empirical amplitude calibration |
| F field solver | partial_repo_module | smoke-scale resistive field solve | calibrated empirical field evidence |
| P probe | prototype_api | developmental laminar readouts | calibrated EEG/MEG evidence |
| A analysis/objective | partial_repo_module | internal motif gate/profile metrics | null-normalized S_lam without null |
| O optimizer | partial_repo_module | bounded parameter search | biological mechanism proof |
| C constraints | partial_repo_module | finite/source/passivity checks | missing evidence implies pass |

Use `from jbiophysic import jtfne; jtfne.status()` for the machine-readable table.
