"""Metrics and calibration.

Classification: per-class sensitivity (recall), PPV (precision), F1, macro-F1
and the confusion matrix, as in the ECG literature (AAMI EC57 reports Se/+P).

Calibration: expected calibration error (top-label, equal-width bins),
reliability diagrams, and temperature scaling (one scalar T fit by
minimizing NLL on validation logits).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


# ---------------------------------------------------------------- classification

def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def class_metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    cm = confusion(y, pred, len(classes))
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(support > 0, tp / support, np.nan)
        ppv = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((se + ppv) > 0, 2 * se * ppv / (se + ppv), 0.0)
    f1 = np.where(support > 0, f1, np.nan)  # undefined without positives
    per_class = {c: {"support": int(support[i]), "sensitivity": _f(se[i]),
                     "ppv": _f(ppv[i]), "f1": _f(f1[i])}
                 for i, c in enumerate(classes)}
    return {"per_class": per_class,
            "macro_f1": _f(np.nanmean(f1)),
            "accuracy": _f(tp.sum() / cm.sum()),
            "confusion": cm.tolist()}


def _f(v) -> float | None:
    return None if v is None or np.isnan(v) else float(v)


# ---------------------------------------------------------------- calibration

def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: weighted mean |accuracy - confidence| over bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append((edges[b], edges[b + 1], int(m.sum()),
                    float(conf[m].mean()) if m.any() else np.nan,
                    float(correct[m].mean()) if m.any() else np.nan))
    return out


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T > 0 minimizing NLL. Fit on validation logits only."""
    res = minimize_scalar(lambda lt: nll(logits, y, np.exp(lt)),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def ensemble_probs(fold_logits: list[np.ndarray], T: float = 1.0) -> np.ndarray:
    """Apply T to each model's logits, softmax, then average probabilities."""
    return np.mean([softmax(lg / T, axis=1) for lg in fold_logits], axis=0)


# ---------------------------------------------------------------- CV report (DS1 only)

def report_cv(cfg: dict, run_name: str) -> dict:
    """Out-of-fold metrics and training curves for a CV run. Reads DS1 only."""
    import json

    import pandas as pd

    from ecg import plotting as P
    from ecg.config import results_dir, run_dir as get_run_dir

    classes = cfg["data"]["classes"]
    run_dir = get_run_dir(cfg, run_name)
    out = results_dir(cfg, run_name)
    out.mkdir(parents=True, exist_ok=True)
    o = np.load(run_dir / "oof.npz")
    summary = json.loads((run_dir / "summary.json").read_text())
    logits, y, fold, record = o["logits"], o["y"], o["fold"], o["record"]
    pred = logits.argmax(1)

    pooled = class_metrics(y, pred, classes)
    probs = softmax(logits, axis=1)
    pooled["ece_uncalibrated"] = ece(probs, y, cfg["calibration"]["n_bins"])
    pooled["nll_uncalibrated"] = nll(logits, y)

    # Per fold. For the patient-level scheme, F is reported only for the fold
    # holding record 208 (90% of DS1's F beats); elsewhere F has 0-21 beats.
    f_fold = int(np.unique(fold[record == 208])[0]) if summary["scheme"] == "inter" else None
    rows = []
    for k in sorted(np.unique(fold)):
        m = fold == k
        cm_k = class_metrics(y[m], pred[m], classes)
        row = {"fold": int(k), "records": " ".join(map(str, sorted(np.unique(record[m])))) if f_fold is not None else "random beats",
               "best_epoch": summary["folds"][k]["best_epoch"]}
        for c in classes:
            pc = cm_k["per_class"][c]
            show = not (c == "F" and f_fold is not None and k != f_fold)
            row[f"{c}_n"] = pc["support"]
            row[f"{c}_Se"] = pc["sensitivity"] if show else None
            row[f"{c}_PPV"] = pc["ppv"] if show else None
        nsv = [cm_k["per_class"][c]["f1"] for c in ("N", "S", "V")]
        row["macroF1_NSV"] = float(np.mean(nsv))
        rows.append(row)
    per_fold = pd.DataFrame(rows).set_index("fold")
    per_fold.to_csv(out / "per_fold_metrics.csv", float_format="%.3f")

    result = {"experiment": cfg["experiment"], "run": run_name, "scheme": summary["scheme"],
              "fixed_epochs": summary["fixed_epochs"],
              "pooled_oof": pooled,
              "per_fold_macroF1_NSV_mean": float(per_fold["macroF1_NSV"].mean()),
              "per_fold_macroF1_NSV_std": float(per_fold["macroF1_NSV"].std(ddof=1)),
              "f_metrics_fold": f_fold}
    (out / "oof_metrics.json").write_text(json.dumps(result, indent=1))
    _plot_curves(run_dir, summary, out / "training_curves.png", run_name, P)
    return result


def _plot_curves(run_dir, summary, path, title, P) -> None:
    import json

    import matplotlib.pyplot as plt

    P.setup()
    k = len(summary["folds"])
    fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 5.6), sharex=True, sharey="row")
    for i, f in enumerate(summary["folds"]):
        h = json.loads((run_dir / f"fold{i}" / "history.json").read_text())
        ep = np.arange(1, len(h["loss"]) + 1)
        a, b = axes[0, i], axes[1, i]
        a.plot(ep, h["loss"], color=P.INK_2, label="train")
        a.plot(ep, h["val_loss"], color="#6250d6", label="validation")
        key = "val_macro_f1_nsv" if "val_macro_f1_nsv" in h else "val_macro_f1"
        b.plot(ep, h[key], color="#6250d6")
        b.set_ylabel("")
        for ax in (a, b):
            ax.axvline(f["best_epoch"], color=P.INK_2, lw=0.8, ls="--")
        a.set_title(f"fold {i} (best epoch {f['best_epoch']})")
        b.set_xlabel("epoch")
    axes[0, 0].set_ylabel("class-weighted loss")
    h0 = json.loads((run_dir / "fold0" / "history.json").read_text())
    axes[1, 0].set_ylabel("val. macro-F1 (N,S,V)" if "val_macro_f1_nsv" in h0 else "val. macro-F1 (N,S,V,F)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"{title}: training curves per fold (dashed = selected epoch)", y=1.01)
    fig.savefig(path)
    plt.close(fig)
