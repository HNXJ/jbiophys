import copy
import hashlib
import json
from pathlib import Path

import yaml


def _load(name):
    return yaml.safe_load(Path("configs", name).read_text())


def _strip_for_invariant(d):
    d = copy.deepcopy(d)
    for k in ["ratio_mode"]:
        d.pop(k, None)
    if "metadata" in d:
        d["metadata"].pop("description", None)
    # The only intended manipulation is init.mode/layer_cell_fractions.
    d.get("init", {}).pop("mode", None)
    d.get("init", {}).pop("layer_cell_fractions", None)
    return d


def _hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def test_correct_inverse_ratio_configs_are_paired_except_allocation():
    c = _load("tfne_spectrolaminar_correct_ratio.yaml")
    i = _load("tfne_spectrolaminar_inverse_ratio.yaml")
    assert c["truth_mode"] == i["truth_mode"] == "truth_safe_unverified"
    assert c["claim_level"] == i["claim_level"] == "controlled_ablation_scaffold"
    assert c["ratio_mode"] == "correct"
    assert i["ratio_mode"] == "inverse"
    assert c["seed_list"] == i["seed_list"]
    assert c["time_window_ms"] == i["time_window_ms"] == [-500.0, 1000.0]
    assert c["event_window_ms"] == i["event_window_ms"]
    assert c["post_window_ms"] == i["post_window_ms"]
    assert c["field_readout"] == i["field_readout"]
    assert c["solver"] == i["solver"]
    assert (
        c["source_calibration_status"]
        == i["source_calibration_status"]
        == "toy_scale_A_per_native_not_empirical"
    )
    assert c["pairing_policy"]["shared_positions"] is True
    assert i["pairing_policy"]["cell_type_allocation_only_differs"] is True
    assert _hash(_strip_for_invariant(c)) == _hash(_strip_for_invariant(i))


def test_layer_totals_match_after_loading_jtfne_configs():
    from jbiophysic import jtfne

    cc = jtfne.load_cfg("configs/tfne_spectrolaminar_correct_ratio.yaml")
    ii = jtfne.load_cfg("configs/tfne_spectrolaminar_inverse_ratio.yaml")
    assert cc.init.layer_fractions == ii.init.layer_fractions
    assert cc.init.area_order == ii.init.area_order
    assert cc.sim.tfne_grid_nxy == ii.sim.tfne_grid_nxy
    assert cc.sim.n_contacts == ii.sim.n_contacts
