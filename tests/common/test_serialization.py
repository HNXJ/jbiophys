import json
from pathlib import Path

import numpy as np
import pandas as pd

from jbiophysic.io.manifests import hash_assets, json_safe, write_json_manifest


def test_manifest_no_nan_inf_json(tmp_path):
    payload = {
        "nan": np.nan,
        "inf": np.inf,
        "ninf": -np.inf,
        "arr": np.asarray([1.0, np.nan, np.inf]),
        "df": pd.DataFrame({"x": [1.0, np.nan]}),
        "path": Path("abc"),
    }
    safe = json_safe(payload)
    text = json.dumps(safe, allow_nan=False)
    assert "NaN" not in text
    assert safe["nan"] is None
    path = tmp_path / "manifest.json"
    write_json_manifest(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded["inf"] is None


def test_hash_assets(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("abc")
    hashes = hash_assets([p, tmp_path / "missing.txt"])
    assert str(p) in hashes
    assert len(hashes[str(p)]) == 64
