"""JSON manifest IO and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively convert common scientific Python objects into strict JSON values.

    Non-finite floats are converted to ``None`` so ``json.dumps(..., allow_nan=False)``
    succeeds.  This is intentionally conservative for manifests used as evidence:
    missing or invalid values are represented as JSON null, never NaN/Infinity.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is core in this project
        np = None
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is core in this project
        pd = None

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if np is not None:
        if isinstance(obj, np.generic):
            return json_safe(obj.item())
        if isinstance(obj, np.ndarray):
            return json_safe(obj.tolist())
    if pd is not None and isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if pd is not None and isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return json_safe(asdict(obj))
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def hash_file(path: str | Path) -> str:
    """Return SHA256 hash of a file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_assets(paths: list[str | Path] | tuple[str | Path, ...]) -> dict[str, str]:
    """Return ``{path: sha256}`` for existing assets."""
    out: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        if path.exists() and path.is_file():
            out[str(path)] = hash_file(path)
    return out


def write_json_manifest(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json_safe(payload)
    path.write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def create_run_manifest(
    seed: int,
    dt_ms: float,
    duration_ms: float,
    model_family: str = "izhikevich",
    current_unit: str = "izhikevich_model_unit",
    nan_inf_status: str = "pass",
    bounds_status: str = "pass",
) -> dict[str, object]:
    """Create a standard run manifest for tutorial provenance."""
    return {
        "truth_status": "truth_safe_unverified",
        "seed": seed,
        "dt_ms": dt_ms,
        "simulation_time_ms": duration_ms,
        "model_family": model_family,
        "current_unit": current_unit,
        "tfne_source_calibration": None,
        "nan_inf_status": nan_inf_status,
        "bounds_status": bounds_status,
    }
