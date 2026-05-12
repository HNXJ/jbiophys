"""Differentiable synchrony objectives for JAX-based biophysical searches."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def synchrony_kappa_objective(
    v_traces: jax.Array,
    target_kappa: float = 0.0,
    no_silence_threshold: float = 1.0,
    no_silence_weight: float = 10.0,
) -> jax.Array:
    """Target a Fleiss-kappa-like synchrony of zero while avoiding silence.

    Args:
        v_traces: [time, units] voltage traces or activity.
        target_kappa: The desired synchrony level (usually 0.0 for balanced activity).
        no_silence_threshold: Minimum average variance/activity required to avoid penalty.
        no_silence_weight: Weight for the no-silence hinge penalty.

    Returns:
        Scalar differentiable loss.
    """
    if v_traces.ndim != 2:
        raise ValueError("v_traces must be [time, units]")

    # Normalize traces to unit variance for correlation-like behavior
    means = jnp.mean(v_traces, axis=0, keepdims=True)
    centered = v_traces - means
    vars = jnp.mean(centered**2, axis=0)
    stds = jnp.sqrt(vars + 1e-8)
    normalized = centered / stds

    # Compute mean pairwise correlation (proxy for kappa)
    # R = (1/T) * (1/N^2) * sum_i,j (v_i * v_j)
    # Which is equivalent to (1/T) * mean( (sum_i v_i)^2 )
    summed_activity = jnp.mean(normalized, axis=1)
    kappa = jnp.mean(summed_activity**2) - (1.0 / v_traces.shape[1])

    # Squared penalty for deviating from target kappa
    sync_loss = jnp.square(kappa - target_kappa)

    # No-silence penalty: ensure some minimum activity (variance)
    mean_var = jnp.mean(vars)
    silence_penalty = no_silence_weight * jnp.maximum(0.0, no_silence_threshold - mean_var)

    return sync_loss + silence_penalty
