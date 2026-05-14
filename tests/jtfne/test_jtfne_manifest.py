import json

from jbiophysic import jtfne


def test_manifest_is_json_safe_and_has_operator_status(tmp_path):
    cfg = jtfne.default_cfg("correct", smoke=True)
    model = jtfne.construct(cfg.init)
    sig = jtfne.simulate(model, cfg.sim)
    ev = jtfne.evaluate(sig, cfg.opt)
    manifest = jtfne.write_manifest(sig, tmp_path, evaluation=ev)
    text = json.dumps(manifest, allow_nan=False)
    assert "operator_status" in manifest
    assert "chemical" in manifest["operator_status"]
    assert "NaN" not in text
    loaded = json.loads((tmp_path / "manifest.json").read_text())
    assert loaded["array_layout"]["current_density"] == "channel_first"
