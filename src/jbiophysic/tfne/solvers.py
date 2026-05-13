"""Small TFNE solver helpers.

These routines are intentionally smoke-test scale. Production elliptic solves should use a
proper gauge-fixed sparse solver or FEM backend.
"""

from __future__ import annotations

import jax.numpy as jnp

from .fields import FieldSolution, TFNEGrid, mean_zero_gauge


def jacobi_poisson_neumann_smoke(
    source: jnp.ndarray,
    grid: TFNEGrid,
    *,
    conductivity_s_m: float = 0.3,
    steps: int = 200,
    residual_tol: float = 1e-6,
) -> FieldSolution:
    """Approximate `sigma * laplacian(phi) = -source` for small smoke tests.

    The source is mean-centered to satisfy a zero-flux Neumann compatibility condition.
    Only uniform spacing is supported.

    Returns a FieldSolution with residual norm and convergence status.
    """
    if conductivity_s_m <= 0:
        raise ValueError("conductivity_s_m must be positive")
    if len(set(float(x) for x in grid.dx)) != 1:
        raise ValueError("smoke Jacobi solver requires equal grid spacing")
    if steps < 1:
        raise ValueError("steps must be positive")
    h = float(grid.dx[0])
    rhs = source - jnp.mean(source)
    phi = jnp.zeros(grid.shape, dtype=source.dtype)

    residual_norm = 0.0
    n_iterations = 0

    for iteration in range(steps):
        p = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")
        neighbor_sum = (
            p[2:, 1:-1, 1:-1]
            + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1]
            + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:]
            + p[1:-1, 1:-1, :-2]
        )
        phi_new = (neighbor_sum + (h**2) * rhs / conductivity_s_m) / 6.0
        phi_new = mean_zero_gauge(phi_new, grid.active_mask)

        # Compute residual: ||Ax - b|| where Ax = sigma * laplacian(phi)
        # For Poisson: laplacian(phi) ≈ (neighbor_sum - 6*phi) / h^2
        # So residual ≈ ||sigma * (neighbor_sum - 6*phi) / h^2 + source||
        neighbor_sum_new = (
            p[2:, 1:-1, 1:-1]
            + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1]
            + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:]
            + p[1:-1, 1:-1, :-2]
        )
        laplacian = (neighbor_sum_new - 6.0 * phi_new) / (h**2)
        residual = conductivity_s_m * laplacian + source
        residual_norm = float(jnp.linalg.norm(residual[grid.active_mask]))

        phi = phi_new
        n_iterations = iteration + 1

        # Early exit if converged
        if residual_norm < residual_tol:
            break

    converged = residual_norm < residual_tol

    return FieldSolution(
        phi_e=phi,
        residual_norm=residual_norm,
        n_iterations=n_iterations,
        converged=converged,
        gauge_applied="mean_zero",
        boundary_condition="neumann_zero",
        solver_name="jacobi_poisson_neumann_smoke",
        claim_level="smoke_test",
    )
