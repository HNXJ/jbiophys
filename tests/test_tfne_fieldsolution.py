"""Tests for TFNE FieldSolution dataclass and solver contract.

Validates that the Jacobi Poisson Neumann solver returns complete provenance
metadata (residual norm, iteration count, convergence status, gauge, boundary
conditions) rather than just a raw field array.
"""

import jax.numpy as jnp
import pytest

from jbiophysic.tfne import FieldSolution, jacobi_poisson_neumann_smoke, make_regular_grid


class TestFieldSolutionDataclass:
    """Test FieldSolution contract and immutability."""

    def test_fieldsolution_is_frozen(self):
        """Verify FieldSolution is frozen (immutable)."""
        phi_e = jnp.zeros((5, 5, 5))
        sol = FieldSolution(
            phi_e=phi_e,
            residual_norm=1.5e-6,
            n_iterations=42,
            converged=True,
            gauge_applied="mean_zero",
        )
        with pytest.raises(Exception):  # dataclass frozen raises AttributeError or TypeError
            sol.residual_norm = 2.0

    def test_fieldsolution_default_values(self):
        """Verify FieldSolution default values."""
        phi_e = jnp.zeros((5, 5, 5))
        sol = FieldSolution(
            phi_e=phi_e,
            residual_norm=1e-6,
            n_iterations=10,
            converged=True,
            gauge_applied="mean_zero",
        )
        assert sol.boundary_condition == "neumann_zero"
        assert sol.solver_name == "jacobi_poisson_neumann_smoke"
        assert sol.claim_level == "smoke_test"

    def test_fieldsolution_custom_values(self):
        """Verify FieldSolution accepts custom values for optional fields."""
        phi_e = jnp.zeros((5, 5, 5))
        sol = FieldSolution(
            phi_e=phi_e,
            residual_norm=1e-6,
            n_iterations=10,
            converged=True,
            gauge_applied="pinned",
            boundary_condition="dirichlet",
            solver_name="custom_solver",
            claim_level="computational",
        )
        assert sol.boundary_condition == "dirichlet"
        assert sol.solver_name == "custom_solver"
        assert sol.claim_level == "computational"


class TestJacobiPoissonNeumannSmokeSolver:
    """Test that jacobi_poisson_neumann_smoke returns FieldSolution with correct metadata."""

    def test_solver_returns_fieldsolution(self):
        """Verify solver returns FieldSolution object, not bare array."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)
        assert isinstance(result, FieldSolution)

    def test_solver_returns_all_required_fields(self):
        """Verify FieldSolution contains all required fields."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert hasattr(result, "phi_e")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "converged")
        assert hasattr(result, "gauge_applied")
        assert hasattr(result, "boundary_condition")
        assert hasattr(result, "solver_name")
        assert hasattr(result, "claim_level")

    def test_phi_e_is_array_with_correct_shape(self):
        """Verify phi_e array has same shape as grid."""
        grid = make_regular_grid(shape=(7, 8, 9), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((7, 8, 9)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert result.phi_e.shape == grid.shape
        assert result.phi_e.ndim == 3

    def test_residual_norm_is_nonnegative_float(self):
        """Verify residual_norm is a non-negative float."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert isinstance(result.residual_norm, (float, jnp.ndarray))
        assert float(result.residual_norm) >= 0.0

    def test_n_iterations_is_positive_int(self):
        """Verify n_iterations is a positive integer."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert isinstance(result.n_iterations, (int, jnp.integer))
        assert int(result.n_iterations) > 0
        assert int(result.n_iterations) <= 10

    def test_converged_bool_matches_residual_tolerance(self):
        """Verify converged flag matches residual_norm < residual_tol."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01

        # Test with tight tolerance (may not converge)
        result_tight = jacobi_poisson_neumann_smoke(
            source, grid, steps=5, residual_tol=1e-10
        )
        # May or may not converge depending on iterations

        # Test with loose tolerance (should converge quickly)
        result_loose = jacobi_poisson_neumann_smoke(
            source, grid, steps=200, residual_tol=1.0
        )
        # Should converge with loose tolerance
        assert result_loose.converged is True
        assert float(result_loose.residual_norm) < 1.0

    def test_gauge_applied_is_mean_zero(self):
        """Verify gauge_applied field is set correctly."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert result.gauge_applied == "mean_zero"

    def test_boundary_condition_is_neumann_zero(self):
        """Verify boundary_condition field is set correctly."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert result.boundary_condition == "neumann_zero"

    def test_solver_name_is_correct(self):
        """Verify solver_name field is set to the actual solver name."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert result.solver_name == "jacobi_poisson_neumann_smoke"

    def test_claim_level_is_smoke_test(self):
        """Verify claim_level field is set to smoke_test."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=10, residual_tol=1e-6)

        assert result.claim_level == "smoke_test"

    def test_solver_early_exits_on_convergence(self):
        """Verify solver stops early if convergence threshold is met."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.001  # Smaller source

        # With very loose tolerance, should converge in few iterations
        result = jacobi_poisson_neumann_smoke(
            source, grid, steps=1000, residual_tol=1.0  # Very loose
        )
        # n_iterations should be much less than 1000
        assert int(result.n_iterations) < 500
        assert result.converged is True

    def test_solver_respects_max_steps(self):
        """Verify solver does not exceed max steps parameter."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01

        max_steps = 50
        result = jacobi_poisson_neumann_smoke(
            source, grid, steps=max_steps, residual_tol=1e-12
        )
        # n_iterations should not exceed max_steps
        assert int(result.n_iterations) <= max_steps

    def test_residual_computation_is_consistent(self):
        """Verify residual norm computation is reasonable."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=200, residual_tol=1e-6)

        # For converged solution, residual should be small
        if result.converged:
            assert float(result.residual_norm) < 1e-6


class TestFieldSolutionInterpretability:
    """Test that FieldSolution enables interpretability of field outputs."""

    def test_field_output_is_interpretable_as_computational_result(self):
        """Verify FieldSolution captures enough metadata for interpretation."""
        grid = make_regular_grid(shape=(5, 5, 5), dx=(0.1, 0.1, 0.1))
        source = jnp.ones((5, 5, 5)) * 0.01
        result = jacobi_poisson_neumann_smoke(source, grid, steps=100, residual_tol=1e-6)

        # Check that result is not just a bare array - it has provenance
        assert hasattr(result, "phi_e"), "Missing phi_e field"
        assert hasattr(result, "residual_norm"), "Missing residual_norm (solver status)"
        assert hasattr(result, "converged"), "Missing converged flag (solution quality)"
        assert hasattr(result, "gauge_applied"), "Missing gauge info"
        assert hasattr(result, "boundary_condition"), "Missing boundary condition info"
        assert hasattr(result, "solver_name"), "Missing solver identification"
        assert hasattr(result, "claim_level"), "Missing claim level (scientific status)"

        # Verify the metadata is meaningful
        assert result.solver_name != "", "Solver name should be non-empty"
        assert result.gauge_applied != "", "Gauge should be specified"
        assert result.boundary_condition != "", "Boundary condition should be specified"
        assert result.claim_level in [
            "computational",
            "smoke_test",
        ], "claim_level should be valid"
