import numpy as np

from jbiophysic import jtfne
from jbiophysic.validation.field_invariants import normalize_current_density_layout


def test_jtfne_validate_accepts_smoke_signals():
    cfg = jtfne.default_cfg("correct", smoke=True)
    model = jtfne.construct(cfg.init)
    sig = jtfne.simulate(model, cfg.sim)
    report = jtfne.validate(sig)
    assert report.accepted, report.failures


def test_current_density_layout_conversion():
    cf = np.zeros((2, 3, 4, 5, 6))
    cl = np.zeros((2, 4, 5, 6, 3))
    assert normalize_current_density_layout(cf, layout="channel_first").shape == cf.shape
    assert normalize_current_density_layout(cl, layout="channel_last").shape == cf.shape
