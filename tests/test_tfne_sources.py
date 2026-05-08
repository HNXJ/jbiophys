import jax.numpy as jnp

from jbiophysic.tfne.fields import make_regular_grid
from jbiophysic.tfne.sources import conservation_error, gaussian_mollifier, project_sparse_currents
from jbiophysic.tfne.validation import assert_source_conserved


def test_gaussian_mollifier_conserves_unit_integral():
    grid = make_regular_grid((7, 7, 7), (20e-6, 20e-6, 20e-6))
    pos = jnp.array([60e-6, 60e-6, 60e-6])
    eta = gaussian_mollifier(grid, pos, radius_m=25e-6)
    integral = jnp.sum(eta * grid.voxel_volume)
    assert abs(float(integral) - 1.0) < 2e-6


def test_sparse_current_projection_conserves_current():
    grid = make_regular_grid((9, 9, 9), (20e-6, 20e-6, 20e-6))
    currents = jnp.array([1e-12, -0.25e-12, 0.5e-12])
    positions = jnp.array([[40e-6, 40e-6, 40e-6], [80e-6, 80e-6, 80e-6], [120e-6, 80e-6, 40e-6]])
    radii = jnp.array([25e-6, 25e-6, 30e-6])
    q = project_sparse_currents(grid, currents, positions, radii)
    target = float(jnp.sum(currents))
    assert abs(float(conservation_error(grid, q, target))) < 1e-18
    assert_source_conserved(grid, q, target, rtol=1e-4)
