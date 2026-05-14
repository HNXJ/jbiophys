import numpy as np

from jbiophysic.objectives.spectrolaminar import (
    compute_motif_vector,
    laminar_similarity,
    make_null_distribution,
    motif_gate_score,
)


def test_motif_gate_and_similarity_are_separated():
    z = np.asarray([-0.5, -0.2, 0.1, 0.4])
    ab = np.asarray([0.1, 0.2, 0.8, 1.0])
    gm = np.asarray([1.0, 0.8, 0.2, 0.1])
    m = compute_motif_vector(ab, gm, z)
    assert motif_gate_score(m) == 100.0
    target = compute_motif_vector(ab, gm, z)
    null = compute_motif_vector(np.ones_like(ab) * 0.5, np.ones_like(gm) * 0.5, z)
    assert laminar_similarity(m, target, null) > 0.99


def test_null_distribution_requires_declared_type():
    z = np.asarray([-0.5, -0.2, 0.1, 0.4])
    profile = {"alpha_beta": np.arange(4.0), "gamma": np.arange(4.0)[::-1], "pos_from_l4": z}
    nulls = make_null_distribution(profile, null_type="layer_shuffle", n=4, seed=1)
    assert len(nulls) == 4
