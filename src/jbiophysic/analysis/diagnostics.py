"""Physiological and run diagnostics for JTFNE/TFNE scaffold outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fleiss_kappa_binary(spikes: np.ndarray, *, bin_ms: float, dt_ms: float) -> float:
    """Fleiss-kappa-like agreement metric for binned population activity."""
    spikes = np.asarray(spikes).astype(bool)
    bin_n = max(1, int(round(bin_ms / dt_ms)))
    nbin = spikes.shape[0] // bin_n
    if nbin < 2 or spikes.shape[1] < 2:
        return 0.0
    b = spikes[: nbin * bin_n].reshape(nbin, bin_n, spikes.shape[1]).any(axis=1).astype(float)
    p = b.mean(axis=1)
    observed = np.mean(p * p + (1.0 - p) * (1.0 - p))
    p_global = b.mean()
    expected = p_global * p_global + (1.0 - p_global) * (1.0 - p_global)
    denom = 1.0 - expected
    return 0.0 if abs(denom) < 1e-12 else float((observed - expected) / denom)


def mean_pairwise_spike_correlation(spikes: np.ndarray) -> float:
    spikes = np.asarray(spikes).astype(float)
    if spikes.shape[1] < 2 or spikes.shape[0] < 2:
        return 0.0
    active = spikes.std(axis=0) > 0
    if active.sum() < 2:
        return 0.0
    corr = np.corrcoef(spikes[:, active].T)
    iu = np.triu_indices(corr.shape[0], k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def burst_index(
    spikes: np.ndarray, *, bin_ms: float, dt_ms: float, active_fraction_threshold: float = 0.50
) -> float:
    spikes = np.asarray(spikes).astype(bool)
    bin_n = max(1, int(round(bin_ms / dt_ms)))
    nbin = spikes.shape[0] // bin_n
    if nbin < 1:
        return 0.0
    b = spikes[: nbin * bin_n].reshape(nbin, bin_n, spikes.shape[1]).any(axis=1)
    pop_active = b.mean(axis=1)
    return float(np.mean(pop_active >= active_fraction_threshold))


def celltype_diagnostics(trials, area_order) -> pd.DataFrame:
    """Return firing/silent/voltage diagnostics by area, layer, and cell type."""
    rows = []
    for trial_idx, tr in enumerate(trials):
        dt_ms = float(tr["dt_ms"])
        for area in area_order:
            data = tr[area]
            neurons = data["neurons"]
            spikes = np.asarray(data["spikes"])
            voltage = np.asarray(data["voltage_mV"])
            rates = spikes.mean(axis=0) * 1000.0 / dt_ms
            for (layer, cell_type), idx in neurons.groupby(["layer", "cell_type"]).groups.items():
                idx = np.asarray(list(idx), dtype=int)
                if idx.size == 0:
                    continue
                rows.append(
                    {
                        "trial": trial_idx,
                        "area": area,
                        "layer": layer,
                        "cell_type": cell_type,
                        "n_cell": int(idx.size),
                        "firing_rate_mean_hz": float(np.mean(rates[idx])),
                        "firing_rate_sd_hz": float(np.std(rates[idx])),
                        "silent_fraction": float(np.mean(rates[idx] <= 1e-9)),
                        "voltage_min_mV": float(np.min(voltage[:, idx])),
                        "voltage_p05_mV": float(np.percentile(voltage[:, idx], 5)),
                        "voltage_p95_mV": float(np.percentile(voltage[:, idx], 95)),
                        "voltage_max_mV": float(np.max(voltage[:, idx])),
                    }
                )
    return pd.DataFrame(rows)


def synchrony_diagnostics(trials, area_order, *, bin_ms: float = 10.0) -> pd.DataFrame:
    rows = []
    for trial_idx, tr in enumerate(trials):
        dt_ms = float(tr["dt_ms"])
        for area in area_order:
            spikes = np.asarray(tr[area]["spikes"])
            rows.append(
                {
                    "trial": trial_idx,
                    "area": area,
                    "fleiss_kappa_proxy": fleiss_kappa_binary(spikes, bin_ms=bin_ms, dt_ms=dt_ms),
                    "mean_pairwise_spike_correlation": mean_pairwise_spike_correlation(spikes),
                    "burst_index": burst_index(spikes, bin_ms=bin_ms, dt_ms=dt_ms),
                }
            )
    return pd.DataFrame(rows)


def area_diagnostics(trials, area_order) -> pd.DataFrame:
    rows = []
    for trial_idx, tr in enumerate(trials):
        for area in area_order:
            data = tr[area]
            rows.append(
                {
                    "trial": trial_idx,
                    "area": area,
                    "N_trial": len(trials),
                    "N_area": len(area_order),
                    "N_cell": int(data["spikes"].shape[1]),
                    "N_contact": int(data["lfp_contacts"].shape[1]),
                    "source_calibration_status": data["metadata"].get(
                        "source_calibration_status", "unknown"
                    ),
                    "source_current_min_proxy": float(np.min(data["csd_contacts"])),
                    "source_current_max_proxy": float(np.max(data["csd_contacts"])),
                }
            )
    return pd.DataFrame(rows)
