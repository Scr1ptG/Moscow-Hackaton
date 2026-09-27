"""Загрузка и инференс моделей из репозитория (без датасета): ловит порчу артефактов, в т.ч. CRLF на Windows."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _frame(feats, n=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 60, size=(n, len(feats))), columns=feats)
    X["tgt_is_trip_start"] = 0
    return X


def test_compact_model_loads_and_explains():
    from ml.predictor import CompactPredictor

    pr = CompactPredictor.load()
    out = pr.predict_frame(_frame(pr.feats))
    assert len(out) == 5 and np.isfinite(out["delay_pred"]).all()
    contrib = out[[c for c in out.columns if c.startswith("contrib_")]].sum(axis=1)
    assert np.allclose(out["base_value"] + contrib + out["rule_contrib"], out["delay_pred"], atol=1e-6)
    assert out["explanation"].str.len().min() > 20


def test_ensemble_model_loads():
    from ml.predictor import DelayPredictor

    pr = DelayPredictor.load()
    out = pr.predict_frame(_frame(pr.feats), None, explain=False)
    assert np.isfinite(out["delay_pred"]).all()
