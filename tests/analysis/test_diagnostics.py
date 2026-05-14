import numpy as np

from jbiophysic.analysis.diagnostics import (
    burst_index,
    fleiss_kappa_binary,
    mean_pairwise_spike_correlation,
)


def test_synchrony_diagnostics_finite():
    spikes = np.zeros((20, 4), dtype=bool)
    spikes[::5, 0] = True
    spikes[1::5, 1] = True
    assert np.isfinite(fleiss_kappa_binary(spikes, bin_ms=1.0, dt_ms=1.0))
    assert np.isfinite(mean_pairwise_spike_correlation(spikes))
    assert 0.0 <= burst_index(spikes, bin_ms=1.0, dt_ms=1.0) <= 1.0
