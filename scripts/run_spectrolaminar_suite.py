#!/usr/bin/env python
"""Deterministic JTFNE spectrolaminar suite runner.

This script converts the notebook-facing workflow into explicit CLI evidence files.
It produces developmental scaffold evidence only; it does not establish biological
mechanism, empirical amplitude calibration, or E/I-ratio necessity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jbiophysic import jtfne
from jbiophysic.io.manifests import hash_assets, write_json_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="force smoke defaults")
    ap.add_argument(
        "--figures", action="store_true", help="render plotly HTML/JSON through jtfne.visualize"
    )
    args = ap.parse_args(argv)

    cfg = jtfne.load_cfg(args.config)
    if args.smoke:
        cfg = cfg.with_smoke_defaults()
    if args.seed is not None:
        from dataclasses import replace

        cfg = replace(cfg, init=replace(cfg.init, seed=args.seed))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = jtfne.construct(cfg.init)
    signals = jtfne.simulate(model, cfg.sim)
    evaluation = jtfne.evaluate(signals, cfg.opt)

    # Core evidence tables.
    evaluation.scores.to_csv(out / "metrics.csv", index=False)
    evaluation.celltype_diagnostics.to_csv(out / "celltype_diagnostics.csv", index=False)
    evaluation.area_diagnostics.to_csv(out / "area_diagnostics.csv", index=False)
    evaluation.synchrony_diagnostics.to_csv(out / "synchrony_diagnostics.csv", index=False)
    # Field invariants per area from basis metadata.
    field_rows = []
    for area, b in signals.model.tfne_basis.items():
        field_rows.append(
            {
                "area": area,
                **{
                    k: v
                    for k, v in b.items()
                    if k not in {"mask", "lfp_basis", "csd_basis", "contact_depths_m"}
                },
            }
        )
    pd.DataFrame(field_rows).to_csv(out / "field_invariants.csv", index=False)
    write_json_manifest(out / "operator_status.json", jtfne.operator_status())

    figures = {}
    if args.figures:
        vis = cfg.vis
        from dataclasses import replace

        vis = replace(
            vis, output_dir=str(out / "figures"), write_json=True, write_html=True, show=False
        )
        figures = jtfne.visualize(signals, vis)
    else:
        (out / "figures").mkdir(exist_ok=True)

    asset_paths = [p for p in out.rglob("*") if p.is_file() and p.name != "asset_hashes.json"]
    asset_hashes = hash_assets(asset_paths)
    write_json_manifest(out / "asset_hashes.json", asset_hashes)
    manifest = jtfne.write_manifest(
        signals, out, evaluation=evaluation, figure_names=[k for k in figures if k != "manifest"]
    )
    manifest["asset_hashes"] = asset_hashes
    write_json_manifest(out / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": "ok", "out": str(out), "manifest": str(out / "manifest.json")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
