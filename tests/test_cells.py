import jax.numpy as jnp

from jbiophysic.cells.hh import simulate_hh
from jbiophysic.cells.izhikevich import IzhikevichParams, simulate_izhikevich
from jbiophysic.models.tfne_izhikevich import IzhikevichTFNEScale, izh_current_to_ampere


def test_izhikevich_smoke_finite_and_spikes():
    current = jnp.ones((400,)) * 10.0
    v, u, spikes = simulate_izhikevich(current, params=IzhikevichParams(), dt_ms=0.25)
    assert bool(jnp.all(jnp.isfinite(v)))
    assert bool(jnp.all(jnp.isfinite(u)))
    assert int(jnp.sum(spikes)) >= 1


def test_hh_smoke_finite():
    current = jnp.ones((1000,)) * 10.0
    v, gates = simulate_hh(current, dt_ms=0.01)
    assert bool(jnp.all(jnp.isfinite(v)))
    assert bool(jnp.all(jnp.isfinite(gates)))
    assert gates.shape == (3, 1000)


def test_izh_tfne_bridge_requires_explicit_scale():
    amps = izh_current_to_ampere(jnp.array([1.0, 2.0]), IzhikevichTFNEScale(1e-12))
    assert abs(float(amps[0]) - 1e-12) < 1e-18
