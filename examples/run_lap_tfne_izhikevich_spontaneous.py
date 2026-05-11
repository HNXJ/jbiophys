#!/usr/bin/env python3
"""Run a LAP-driven Izhikevich spontaneous baseline scaffold.

This example runs spontaneous baseline activity only. It does not include sensory
input, omission input, top-down prediction input, TFNE source projection, CSD/LFP
amplitude validation, or biological mechanism validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jbiophysic.models.lap_izhikevich_baseline import (
    LAPBaselineConfig,
    build_lap_population,
    build_sparse_baseline_weights,
    run_lap_spontaneous_baseline,
    summarize_lap_baseline,
    write_lap_baseline_outputs,
)


def _parse_areas(text: str | None) -> tuple[str, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(part.strip() for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True, help="Path to LAP_data_struct.mat")
    parser.add_argument("--out", type=Path, default=Path("outputs/lap_tfne_izhikevich_spontaneous"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--neurons-per-area", type=int, default=100)
    parser.add_argument("--t-ms", type=float, default=1000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true", help="Use short/small smoke settings")
    parser.add_argument("--areas", default=None, help="Comma-separated area subset, e.g. V1,V2,V4,MT,PFC")
    parser.add_argument("--write-html-anatomy", action="store_true", help="Optionally write Plotly 3-D anatomy HTML if viz dependencies are installed")
    args = parser.parse_args()

    neurons_per_area = 30 if args.quick else args.neurons_per_area
    t_ms = 250.0 if args.quick else args.t_ms
    dt_ms = 0.2 if args.quick else args.dt_ms
    mean_in_degree = min(20, max(neurons_per_area - 1, 1)) if args.quick else 40

    cfg = LAPBaselineConfig(
        mat_path=args.mat,
        seed=args.seed,
        t_ms=t_ms,
        dt_ms=dt_ms,
        neurons_per_area=neurons_per_area,
        include_areas=_parse_areas(args.areas),
        mean_in_degree=mean_in_degree,
    )

    pop = build_lap_population(cfg)
    weights = build_sparse_baseline_weights(pop, cfg)
    result = run_lap_spontaneous_baseline(pop, weights, cfg)
    written = write_lap_baseline_outputs(args.out, pop, weights, result, cfg)

    if args.write_html_anatomy:
        try:
            from jbiophysic.viz.network3d import visualize_network_3d
        except Exception as exc:  # pragma: no cover - optional runtime path.
            raise ImportError(
                "Plotly anatomy output requires jbiophysic.viz.network3d and Plotly. "
                "Install the repo visualization extra."
            ) from exc
        visualize_network_3d(
            pop,
            output_html=args.out / "lap_spontaneous_anatomy.html",
            title="LAP-driven Izhikevich spontaneous baseline anatomy",
            show_column_shells=True,
            show_layers=True,
        )
        written["lap_spontaneous_anatomy.html"] = str(args.out / "lap_spontaneous_anatomy.html")

    summary = summarize_lap_baseline(pop, result, cfg)
    print(f"Wrote summary: {written['summary.json']}")
    print(json.dumps({
        "truth_status": summary["truth_status"],
        "baseline_mode": summary["baseline_mode"],
        "total_neurons": summary["total_neurons"],
        "spike_floor": summary["spike_floor"],
        "rates_hz_by_marker": summary["rates_hz_by_marker"],
    }, indent=2))


if __name__ == "__main__":
    main()
