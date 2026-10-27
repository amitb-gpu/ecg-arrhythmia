"""Unit tests for the parts that can silently go wrong: RR features, beat
drops, patient-level fold separation, metrics, ECE and temperature fitting.

Run: conda run -n ecg python -m pytest -q
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from ecg import data as D
from ecg import evaluate as E
from ecg.config import ROOT, load_config


def test_rr_features_values():
    s = np.array([0, 360, 720, 900, 1440])  # 1.0, 1.0, 0.5 (premature), 1.5 s
    rr = D.rr_features(s, local_beats=10)
    assert np.isnan(rr[0, 0]) and np.isnan(rr[-1, 1])
    i = 3  # the premature beat
    pre, post, local, pre_over_local, post_over_pre = rr[i]
    assert pre == pytest.approx(0.5) and post == pytest.approx(1.5)
    assert local == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert pre_over_local == pytest.approx(0.5 / local)
    assert post_over_pre == pytest.approx(3.0)


def test_local_rr_uses_only_preceding_beats():
    s = np.cumsum(np.r_[0, np.full(20, 360)])
    s2 = s.copy()
    s2[-1] += 500  # change the future: local average for earlier beats must not move
    a, b = D.rr_features(s), D.rr_features(s2)
    np.testing.assert_allclose(a[1:-1, 2], b[1:-1, 2])


def test_edge_and_first_last_beats_dropped():
    rec = D.Record(0, np.empty(0), np.array([10, 500, 1000, 1500, 1990]),
                   np.array(list("NNVNN")), 2000)
    m = D.beat_keep_mask(rec, pre=90, post=162)
    assert m["edge"].tolist() == [True, False, False, False, True]
    assert m["keep"].tolist() == [False, True, True, True, False]


def test_aami_mapping():
    assert D.aami_labels(np.array(list("NLRejAaJSVEF/fQ"))).tolist() == \
        list("NNNNNSSSSVVFQQQ")


def test_rr_scaler_is_fit_on_given_rows_only():
    rng = np.random.default_rng(0)
    tr, va = rng.uniform(0.5, 1.5, (100, 5)), rng.uniform(2, 3, (50, 5))
    sc = D.RRScaler().fit(tr)
    np.testing.assert_allclose(sc.transform(tr).mean(0), 0, atol=1e-5)
    assert (sc.transform(va).mean(0) > 1).all()  # val not re-centred


@pytest.mark.skipif(not (ROOT / "cache" / "folds.json").exists(), reason="dataset not built")
def test_inter_folds_are_patient_disjoint_and_cover_ds1():
    cfg = load_config()
    folds = D.load_folds(cfg, ROOT)
    assert sorted(folds) == sorted(D.DS1)
    assert not set(folds) & set(D.DS2)
    d = D.load_split(cfg, ROOT, "ds1")
    from ecg.train import fold_ids
    ids = fold_ids(cfg, d, "inter")
    for k in range(cfg["cv"]["n_folds"]):
        assert not set(d["record"][ids == k]) & set(d["record"][ids != k])
    assert (d["y"] != D.Q_CODE).all()


def test_class_metrics_toy():
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 0, 2])
    m = E.class_metrics(y, p, ["a", "b", "c"])
    assert m["per_class"]["a"]["sensitivity"] == pytest.approx(2 / 3)
    assert m["per_class"]["a"]["ppv"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["f1"] == pytest.approx(1.0)
    assert np.array(m["confusion"]).sum() == 6


def test_ece_zero_when_calibrated_and_temperature_recovered():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, (20000, 4))
    y = np.array([rng.choice(4, p=p) for p in softmax(true_logits, 1)])
    assert E.ece(softmax(true_logits, 1), y) < 0.02
    overconfident = true_logits * 3.0
    T = E.fit_temperature(overconfident, y)
    assert T == pytest.approx(3.0, rel=0.1)
    assert E.ece(softmax(overconfident / T, 1), y) < E.ece(softmax(overconfident, 1), y)


def test_sqrt_balanced_weights_bounded_and_mean_one():
    from ecg.train import class_weights
    y = np.r_[np.zeros(37064, int), np.ones(830, int), np.full(2573, 2), np.full(37, 3)]  # Run 1 fold-0 training counts
    wb = class_weights(y, 4, "balanced")
    ws = class_weights(y, 4, "sqrt_balanced")
    assert wb[3] / wb[0] == pytest.approx(1002, rel=0.01)
    assert ws[3] / ws[0] == pytest.approx(np.sqrt(1002), rel=0.01)
    assert ws[y].mean() == pytest.approx(1.0)
