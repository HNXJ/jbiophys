"""Spectrolaminar motif metrics and null controls.

These utilities deliberately separate an internal motif gate from continuous or
null-normalized similarity.  ``S_lam`` should only be reported when a declared
null motif vector is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MotifVector:
    deep_alpha_beta: float
    superficial_alpha_beta: float
    deep_gamma: float
    superficial_gamma: float
    z_cross: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.deep_alpha_beta,
                self.superficial_alpha_beta,
                self.deep_gamma,
                self.superficial_gamma,
                self.z_cross,
            ],
            dtype=float,
        )


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def compute_motif_vector(
    alpha_beta_profile, gamma_profile, depths, *, l4_depth: float = 0.0
) -> MotifVector:
    """Compute declared spectrolaminar motif vector.

    Depths are relative to L4; negative is superficial and positive is deep.
    """
    ab = np.asarray(alpha_beta_profile, dtype=float)
    gm = np.asarray(gamma_profile, dtype=float)
    z = np.asarray(depths, dtype=float)
    if ab.shape != gm.shape or ab.shape != z.shape:
        raise ValueError("alpha_beta_profile, gamma_profile, and depths must have same shape")
    superficial = z < l4_depth
    deep = z > l4_depth
    diff = ab - gm
    # Crossing estimate: nearest zero or L4 if degenerate.
    if diff.size == 0 or not np.any(np.isfinite(diff)):
        z_cross = float(l4_depth)
    else:
        idx = int(np.nanargmin(np.abs(diff)))
        z_cross = float(z[idx])
    return MotifVector(
        deep_alpha_beta=_safe_mean(ab[deep]),
        superficial_alpha_beta=_safe_mean(ab[superficial]),
        deep_gamma=_safe_mean(gm[deep]),
        superficial_gamma=_safe_mean(gm[superficial]),
        z_cross=z_cross,
    )


def motif_gate_score(m_model: MotifVector, criteria: dict[str, float] | None = None) -> float:
    """Return internal gate satisfaction in percent.

    Gates: deep alpha/beta > superficial alpha/beta, superficial gamma > deep gamma,
    and crossing near L4.  This is not a null-normalized empirical statistic.
    """
    criteria = criteria or {}
    min_contrast = float(criteria.get("min_contrast", 0.0))
    max_l4_cross_abs = float(criteria.get("max_l4_cross_abs", 0.25))
    gates = [
        m_model.deep_alpha_beta > m_model.superficial_alpha_beta + min_contrast,
        m_model.superficial_gamma > m_model.deep_gamma + min_contrast,
        abs(m_model.z_cross) <= max_l4_cross_abs,
    ]
    return 100.0 * float(sum(gates)) / float(len(gates))


def laminar_similarity(
    m_model: MotifVector, m_target: MotifVector, m_null: MotifVector, eps: float = 1e-12
) -> float:
    """Return null-normalized laminar similarity ``S_lam``.

    Higher is better.  This function requires an explicit null vector; callers should
    not label internal scores as ``S_lam`` without a null.
    """
    model = m_model.as_array()
    target = m_target.as_array()
    null = m_null.as_array()
    denom = np.linalg.norm(null - target) + eps
    return float(1.0 - np.linalg.norm(model - target) / denom)


def make_null_distribution(
    profile: dict[str, np.ndarray],
    *,
    null_type: Literal[
        "layer_shuffle", "phase_randomized", "band_label_shuffle", "uniform_gain"
    ] = "layer_shuffle",
    n: int = 32,
    seed: int = 0,
) -> list[MotifVector]:
    """Create simple motif-vector nulls from already-computed profiles."""
    rng = np.random.default_rng(seed)
    ab = np.asarray(profile["alpha_beta"], dtype=float)
    gm = np.asarray(profile["gamma"], dtype=float)
    z = np.asarray(profile["pos_from_l4"], dtype=float)
    out: list[MotifVector] = []
    for _ in range(n):
        if null_type == "layer_shuffle":
            perm = rng.permutation(len(z))
            out.append(compute_motif_vector(ab[perm], gm[perm], z))
        elif null_type == "band_label_shuffle":
            out.append(compute_motif_vector(gm, ab, z))
        elif null_type == "uniform_gain":
            out.append(
                compute_motif_vector(
                    np.full_like(ab, np.mean(ab)), np.full_like(gm, np.mean(gm)), z
                )
            )
        elif null_type == "phase_randomized":
            # Profile-level proxy: random circular roll preserves values
            # but disrupts laminar alignment.
            shift = int(rng.integers(0, len(z))) if len(z) else 0
            out.append(compute_motif_vector(np.roll(ab, shift), np.roll(gm, -shift), z))
        else:
            raise ValueError(f"unknown null_type: {null_type}")
    return out
