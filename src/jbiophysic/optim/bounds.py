"""Optimizer bounds and parameter transforms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class Bound:
    """Immutable bound range with PyTree-safe clipping."""

    def __init__(self, lower: float, upper: float):
        if not lower < upper:
            raise ValueError(f"lower ({lower}) must be less than upper ({upper})")
        self.lower = lower
        self.upper = upper

    def clip(self, x: Any) -> Any:
        """Clip a JAX array or PyTree to [lower, upper]."""
        return jax.tree.map(lambda leaf: jnp.clip(leaf, self.lower, self.upper), x)

    def __repr__(self) -> str:
        return f"Bound(lower={self.lower}, upper={self.upper})"


def sigmoid_bounded(u: jnp.ndarray, bound: Bound) -> jnp.ndarray:
    """Map unconstrained u to [bound.lower, bound.upper] via sigmoid."""
    return bound.lower + (bound.upper - bound.lower) * jax.nn.sigmoid(u)


def positive_softplus(u: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Map unconstrained u to (eps, inf) via softplus."""
    return jax.nn.softplus(u) + eps
