#!/usr/bin/env python
"""Render or hash spectrolaminar figures for an existing run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from jbiophysic.io.manifests import hash_assets, write_json_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    run = Path(args.run)
    figs = sorted((run / "figures").glob("*")) if (run / "figures").exists() else []
    hashes = hash_assets([*figs, *run.glob("*.csv"), *run.glob("*.json")])
    write_json_manifest(run / "asset_hashes.json", hashes)
    print(f"hashed {len(hashes)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
