"""Validation helpers for TFNE smoke tests."""

from __future__ import annotations

import jax.numpy as jnp

from .fields import TFNEGrid
from .sources import integrate_source
from .tensors import tensor_eigenvalue_diagnostics


def assert_no_nan_inf(name: str, arr: jnp.ndarray) -> None:
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise AssertionError(f"{name} contains NaN/Inf")


def assert_source_conserved(
    grid: TFNEGrid,
    q_A_per_m3: jnp.ndarray,
    target_A: float,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-15,
) -> None:
    got = float(integrate_source(grid, q_A_per_m3))
    if abs(got - target_A) > atol + rtol * max(abs(target_A), atol):
        raise AssertionError(f"source conservation failed: got={got}, target={target_A}")


def assert_passive_spd(Gamma: jnp.ndarray, *, min_eig_floor: float = 0.0) -> None:
    min_eig, _max_eig, cond = tensor_eigenvalue_diagnostics(Gamma)
    if float(min_eig) <= min_eig_floor:
        raise AssertionError(f"tensor is not sufficiently SPD: min_eig={float(min_eig)}")
    if not bool(jnp.isfinite(cond)):
        raise AssertionError("tensor condition number is not finite")


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return float(jnp.asarray(x))


def validate_shape_contract(
    signals: dict[str, object], expected: dict[str, tuple[int, ...]]
) -> dict[str, object]:
    """Validate named array shapes against exact expected tuples.

    Shape contracts should be explicit in manifests. This helper deliberately does
    not infer missing dimensions: missing or mismatched arrays are returned as failures.
    """
    failures: list[str] = []
    observed: dict[str, tuple[int, ...]] = {}
    for name, shape in expected.items():
        if name not in signals:
            failures.append(f"missing {name}")
            continue
        got = tuple(jnp.asarray(signals[name]).shape)
        observed[name] = got
        if tuple(shape) != got:
            failures.append(f"{name}: expected {tuple(shape)}, got {got}")
    return {"accepted": not failures, "failures": failures, "observed_shapes": observed}


def conductivity_diagnostics(Gamma: jnp.ndarray) -> dict[str, float]:
    """Return JSON-safe conductivity symmetry/eigenvalue diagnostics."""
    if Gamma.shape[:2] != (3, 3):
        raise ValueError("Gamma must have leading shape (3, 3)")
    sym_err = jnp.max(jnp.abs(Gamma - jnp.swapaxes(Gamma, 0, 1)))
    min_eig, max_eig, cond = tensor_eigenvalue_diagnostics(Gamma)
    return {
        "conductivity_min_eigenvalue": _as_float(min_eig),
        "conductivity_max_eigenvalue": _as_float(max_eig),
        "conductivity_condition_number": _as_float(cond),
        "conductivity_symmetric_error": _as_float(sym_err),
    }


def validate_field_run(
    grid: TFNEGrid,
    q_A_per_m3: jnp.ndarray,
    phi_e: jnp.ndarray,
    J_e: jnp.ndarray | None = None,
    Gamma: jnp.ndarray | None = None,
    *,
    boundary_condition: str = "declared",
    gauge_type: str = "mean_zero",
    source_projection_mode: str = "declared",
    eps: float = 1e-30,
) -> dict[str, object]:
    """Return JSON-safe physical-invariant diagnostics for a TFNE field run."""
    q = jnp.asarray(q_A_per_m3)
    phi = jnp.asarray(phi_e)
    eps_q = integrate_source(grid, q)
    gauge_residual_abs = jnp.abs(jnp.mean(phi)) if gauge_type == "mean_zero" else jnp.asarray(0.0)
    out: dict[str, object] = {
        "epsilon_q_max": abs(_as_float(eps_q)),
        "epsilon_q_mean": abs(_as_float(eps_q)),
        "epsilon_compat_max": abs(_as_float(eps_q)),
        "gauge_type": gauge_type,
        "gauge_residual_abs": _as_float(gauge_residual_abs),
        "boundary_condition": boundary_condition,
        "source_projection_mode": source_projection_mode,
    }
    if J_e is not None:
        from .csd import divergence_neumann_zero

        csd = divergence_neumann_zero(jnp.asarray(J_e), grid)
        num = jnp.linalg.norm((csd - q).ravel())
        den = jnp.linalg.norm(q.ravel()) + eps
        out["solver_residual_l2_relative"] = _as_float(num / den)
    else:
        out["solver_residual_l2_relative"] = None
    if Gamma is not None:
        out.update(conductivity_diagnostics(jnp.asarray(Gamma)))
    return out
