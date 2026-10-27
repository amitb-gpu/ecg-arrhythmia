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
    return metrics_from_cm(confusion(y, pred, len(classes)), classes)


def metrics_from_cm(cm: np.ndarray, classes: list[str]) -> dict:
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


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean squared distance to the one-hot label."""
    return float(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1).mean())


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


# ---------------------------------------------------------------- calibration variants
# (a) raw, (b) temperature, (c) prior correction, (d) prior correction + temperature.
# Class-weighted training shifts the output toward rare classes:
# p_w(c|x) ∝ w_c p(c|x). Subtracting log w_c from the logits undoes that shift
# (back to the training-fold prior); a temperature only rescales confidence.

VARIANTS = ["raw", "temp", "prior", "prior_temp"]
VARIANT_LABELS = {"raw": "(a) raw", "temp": "(b) temperature",
                  "prior": "(c) prior correction", "prior_temp": "(d) prior + temperature"}


def adjust_logits(logits: np.ndarray, log_w: np.ndarray, variant: str, cal: dict) -> np.ndarray:
    """log_w: log class weights of the training fold that produced these logits
    (one vector, or one row per beat for pooled out-of-fold logits)."""
    z = logits - log_w if variant.startswith("prior") else logits
    if variant == "temp":
        z = z / cal["T"]
    elif variant == "prior_temp":
        z = z / cal["T_prior"]
    return z


def fit_calibration(cfg: dict, run_name: str) -> dict:
    """T and T' from the run's pooled out-of-fold logits (DS1 only)."""
    import json

    from ecg.config import run_dir as get_run_dir

    rd = get_run_dir(cfg, run_name)
    o = np.load(rd / "oof.npz")
    summary = json.loads((rd / "summary.json").read_text())
    log_w = np.log(np.array([f["class_weights"] for f in summary["folds"]]))  # (K, C)
    y, lg = o["y"], o["logits"].astype(np.float64)
    return {"T": fit_temperature(lg, y),
            "T_prior": fit_temperature(lg - log_w[o["fold"]], y),
            "log_w": log_w}


def _prob_metrics(probs: np.ndarray, y: np.ndarray, classes: list[str], n_bins: int) -> dict:
    m = class_metrics(y, probs.argmax(1), classes)
    m["ece"] = ece(probs, y, n_bins)
    m["nll"] = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
    m["brier"] = brier(probs, y)
    return m


def oof_calibration(cfg: dict, run_name: str, cal: dict) -> dict:
    """Variants (a)-(d) on pooled DS1 out-of-fold logits (single models)."""
    from ecg.config import run_dir as get_run_dir

    o = np.load(get_run_dir(cfg, run_name) / "oof.npz")
    lw = cal["log_w"][o["fold"]]
    out, probs = {}, {}
    for v in VARIANTS:
        probs[v] = softmax(adjust_logits(o["logits"].astype(np.float64), lw, v, cal), axis=1)
        out[v] = _prob_metrics(probs[v], o["y"], cfg["data"]["classes"], cfg["calibration"]["n_bins"])
    return {"metrics": out, "probs": probs, "y": o["y"]}


def evaluate_model_set(L: np.ndarray, y: np.ndarray, cal: dict, classes: list[str],
                       n_bins: int) -> dict:
    """L: (K, n, C) logits of K fold models on the same beats.
    Ensemble = average of per-model softmax after each model's adjustment."""
    out = {"ensemble": {}, "single": {}, "probs": {}, "y": y}
    for v in VARIANTS:
        P = np.stack([softmax(adjust_logits(L[k], cal["log_w"][k], v, cal), axis=1)
                      for k in range(len(L))])
        out["ensemble"][v] = _prob_metrics(P.mean(0), y, classes, n_bins)
        singles = [_prob_metrics(P[k], y, classes, n_bins) for k in range(len(L))]
        out["single"][v] = {"models": singles,
                            "mean": _summarize(singles, np.mean), "sd": _summarize(singles, np.std)}
        out["probs"][v] = {"ensemble": P.mean(0), "single_pooled": P.reshape(-1, P.shape[2])}
    return out


def _flat(m: dict) -> dict:
    f = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    for c, pc in m["per_class"].items():
        for k in ("sensitivity", "ppv", "f1"):
            f[f"{c}_{k}"] = pc[k]
    for k in ("ece", "nll", "brier"):
        if k in m:
            f[k] = m[k]
    return f


def _summarize(ms: list[dict], fn) -> dict:
    flat = [_flat(m) for m in ms]
    return {k: float(fn([np.nan if f[k] is None else f[k] for f in flat])) for k in flat[0]}


def bootstrap_by_record(y, pred, record, classes, n_boot=2000, seed=0) -> dict:
    """95% percentile intervals, resampling records (patients) with replacement."""
    recs = np.unique(record)
    cms = np.stack([confusion(y[record == r], pred[record == r], len(classes)) for r in recs])
    draws = np.random.default_rng(seed).integers(0, len(recs), (n_boot, len(recs)))
    with np.errstate(invalid="ignore"):
        stats = [_flat(metrics_from_cm(cms[i].sum(0), classes)) for i in draws]
    ci = {}
    for k in stats[0]:
        v = np.array([np.nan if s[k] is None else s[k] for s in stats], float)
        ok = np.isfinite(v)
        ci[k] = {"lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)),
                 "n_defined": int(ok.sum())} if ok.any() else None
    no_f = int(sum(cms[i].sum(0)[classes.index("F")].sum() == 0 for i in draws)) if "F" in classes else 0
    return {"n_boot": n_boot, "seed": seed, "n_records": int(len(recs)),
            "resamples_without_F": no_f, "ci": ci}


# ---------------------------------------------------------------- test-set evaluation (DS2)

FROZEN_COMMIT = "1e35d94"
FROZEN_FILES = ["configs/default.yaml", "ecg/data.py", "ecg/model.py", "ecg/train.py"]
MODEL_SETS = {  # run name -> role (pre-declared in docs/decisions.md)
    "inter": "primary",
    "inter_fixed12": "secondary (comparison only)",
    "intra_ds1": "diagnostic (beat-level CV models)",
}


def _git(*args) -> str:
    import subprocess

    from ecg.config import ROOT
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def check_frozen(require_clean: bool) -> dict:
    import subprocess

    from ecg.config import ROOT
    changed = subprocess.run(["git", "diff", "--quiet", FROZEN_COMMIT, "--", *FROZEN_FILES], cwd=ROOT).returncode
    if changed:
        raise SystemExit(f"Frozen files differ from {FROZEN_COMMIT}: {FROZEN_FILES}")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if require_clean and dirty:
        raise SystemExit("Working tree has uncommitted changes; commit before evaluating DS2.\n" + dirty)
    return {"frozen_commit": FROZEN_COMMIT, "frozen_files_identical": True,
            "head": _git("rev-parse", "--short", "HEAD"), "clean_tree": not dirty}


def predict_models(cfg: dict, run_name: str, d: dict) -> np.ndarray:
    """(K, n, C) float64 logits; each fold model uses its own training-fold RR scaler."""
    import json

    import keras

    from ecg.config import run_dir as get_run_dir

    out = []
    for k in range(cfg["cv"]["n_folds"]):
        fd = get_run_dir(cfg, run_name) / f"fold{k}"
        sc = json.loads((fd / "scaler.json").read_text())
        rr = ((np.log(d["rr"]) - np.array(sc["mean"])) / np.array(sc["std"])).astype(np.float32)
        model = keras.models.load_model(fd / "best.keras", compile=False)  # inference only
        out.append(model.predict((d["x"][..., None], rr), batch_size=4096, verbose=0).astype(np.float64))
        del model
        keras.backend.clear_session()
    return np.stack(out)


def check_reproduces_oof(cfg: dict, run_name: str, d1: dict) -> float:
    """Reload each fold model, re-predict its validation beats and compare to the
    stored out-of-fold logits: checks loading, scaler and input plumbing."""
    from ecg.config import run_dir as get_run_dir

    worst = 0.0
    L = predict_models(cfg, run_name, d1)
    for k in range(cfg["cv"]["n_folds"]):
        v = np.load(get_run_dir(cfg, run_name) / f"fold{k}" / "val_logits.npz")
        worst = max(worst, float(np.abs(L[k][v["idx"]] - v["logits"]).max()))
    return worst


def evaluate_test(cfg: dict, dry_run: bool = False, overwrite: bool = False, n_boot: int = 2000) -> dict:
    """The single DS2 evaluation. --dry-run runs the identical code on DS1 (where
    every beat was in training for 4 of 5 models, so its numbers mean nothing)
    to exercise the pipeline before DS2 is opened."""
    import datetime
    import json

    import pandas as pd

    from ecg import data as D
    from ecg.config import ROOT, results_dir, run_dir as get_run_dir
    from ecg.train import setup_runtime

    classes, n_bins = cfg["data"]["classes"], cfg["calibration"]["n_bins"]
    split = "ds1" if dry_run else "ds2"
    out = get_run_dir(cfg, "ds2_dryrun") if dry_run else results_dir(cfg, "ds2")
    logit_dir = get_run_dir(cfg, "ds2_dryrun" if dry_run else "ds2")
    if not dry_run and (out / "ds2_metrics.json").exists() and not overwrite:
        raise SystemExit(f"{out}/ds2_metrics.json exists: DS2 has already been evaluated. "
                         "Pass --overwrite only to reproduce the frozen result.")
    # The one real evaluation needs a clean tree; a reproduction (--overwrite, as in
    # scripts/run_all.sh) only needs the frozen files unchanged.
    provenance = check_frozen(require_clean=not (dry_run or overwrite))
    provenance["reproduction"] = bool(overwrite)
    provenance["split"] = split
    provenance["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    setup_runtime(cfg)
    out.mkdir(parents=True, exist_ok=True)
    logit_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        d1 = D.load_split(cfg, ROOT, "ds1", drop_q=True)
        provenance["oof_reproduction_max_abs_diff"] = {
            r: check_reproduces_oof(cfg, r, d1) for r in MODEL_SETS}
        print("re-predicted OOF logits, max |diff|:", provenance["oof_reproduction_max_abs_diff"], flush=True)

    d = D.load_split(cfg, ROOT, split, drop_q=True)
    y, record = d["y"], d["record"]
    result = {"provenance": provenance, "n_beats": int(len(y)),
              "support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
              "roles": MODEL_SETS, "runs": {}}
    sets = {}
    for run in MODEL_SETS:
        cal = fit_calibration(cfg, run)   # DS1 out-of-fold only
        L = predict_models(cfg, run, d)
        np.savez(logit_dir / f"logits_{run}.npz", logits=L, y=y, record=record, sample=d["sample"])
        ev = evaluate_model_set(L, y, cal, classes, n_bins)
        sets[run] = (cal, ev)
        result["runs"][run] = {
            "calibration_params": {"T": cal["T"], "T_prior": cal["T_prior"],
                                   "class_weights": np.exp(cal["log_w"]).tolist()},
            "ensemble": ev["ensemble"],
            "single": {v: {"mean": s["mean"], "sd": s["sd"], "per_model": [_flat(m) for m in s["models"]]}
                       for v, s in ev["single"].items()},
            "bootstrap_ensemble_raw": bootstrap_by_record(
                y, ev["probs"]["raw"]["ensemble"].argmax(1), record, classes, n_boot, cfg["seed"]),
        }
        print(f"[{split}] {run}: ensemble raw macro-F1 "
              f"{ev['ensemble']['raw']['macro_f1']:.3f}", flush=True)

    # DS1 out-of-fold calibration for the primary run, for the side-by-side table.
    oof = oof_calibration(cfg, "inter", sets["inter"][0])
    result["oof_calibration_inter"] = oof["metrics"]

    # Per-record breakdown of the primary result.
    pred = sets["inter"][1]["probs"]["raw"]["ensemble"].argmax(1)
    rows = []
    for r in np.unique(record):
        m = record == r
        row = {"record": int(r), "beats": int(m.sum())}
        for i, c in enumerate(classes):
            n = int((y[m] == i).sum())
            row[f"{c}_n"] = n
            row[f"{c}_Se"] = float((pred[m][y[m] == i] == i).mean()) if n else None
            row[f"pred_{c}"] = int((pred[m] == i).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{split}_per_record.csv", index=False, float_format="%.3f")

    (out / f"{split}_metrics.json").write_text(json.dumps(result, indent=1))
    _plot_confusion(result["runs"]["inter"]["ensemble"]["raw"]["confusion"], classes,
                    out / f"confusion_{split}.png", f"{split.upper()} confusion matrix: primary ensemble, raw argmax")
    _plot_reliability(oof, sets["inter"][1], n_bins, out / f"reliability_{split}.png", split.upper())
    (out / f"{split}_summary.md").write_text(test_summary_md(result, split))
    print((out / f"{split}_summary.md").read_text())
    return result


# ---------------------------------------------------------------- tables and figures

def _fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def test_summary_md(res: dict, split: str) -> str:
    classes = list(res["support"])
    S = split.upper()
    L = [f"## {S} evaluation (frozen config {res['provenance']['frozen_commit']}, run from {res['provenance']['head']}, {res['provenance']['utc']})",
         "", f"{res['n_beats']:,} beats: " + ", ".join(f"{c} {n:,}" for c, n in res["support"].items()), "",
         f"### Raw argmax, 5-model ensembles (95% CI: bootstrap over {S} records)", "",
         "| | " + " | ".join(f"{r} ({role.split(' ')[0]})" for r, role in res["roles"].items()) + " |",
         "|---" * (len(res["roles"]) + 1) + "|"]

    def cell(run, key, getter):
        e = res["runs"][run]["ensemble"]["raw"]
        ci = res["runs"][run]["bootstrap_ensemble_raw"]["ci"].get(key)
        s = _fmt(getter(e))
        return s + (f" [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]" if ci else "")

    L.append("| macro-F1 (N,S,V,F) | " + " | ".join(cell(r, "macro_f1", lambda e: e["macro_f1"]) for r in res["roles"]) + " |")
    for c in classes:
        for k, lab in (("sensitivity", "Se"), ("ppv", "PPV"), ("f1", "F1")):
            L.append(f"| {c} {lab} | " + " | ".join(
                cell(r, f"{c}_{k}", lambda e, c=c, k=k: e["per_class"][c][k]) for r in res["roles"]) + " |")
    L.append("| accuracy | " + " | ".join(_fmt(res["runs"][r]["ensemble"]["raw"]["accuracy"]) for r in res["roles"]) + " |")

    s = res["runs"]["inter"]["single"]["raw"]
    L += ["", "### Primary run: single fold models vs ensemble (raw argmax)", "",
          "| | single models, mean ± sd (n=5) | ensemble |", "|---|---|---|"]
    e = _flat(res["runs"]["inter"]["ensemble"]["raw"])
    for k in ["macro_f1"] + [f"{c}_{m}" for c in classes for m in ("sensitivity", "ppv")]:
        L.append(f"| {k} | {_fmt(s['mean'][k])} ± {_fmt(s['sd'][k])} | {_fmt(e[k])} |")

    cm = res["runs"]["inter"]["ensemble"]["raw"]["confusion"]
    L += ["", "### Primary confusion matrix (rows = true, columns = predicted)", "",
          "| | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    L += [f"| {c} | " + " | ".join(f"{v:,}" for v in cm[i]) + " |" for i, c in enumerate(classes)]

    cp = res["runs"]["inter"]["calibration_params"]
    L += ["", f"### Calibration, primary run (T = {cp['T']:.3f}, T' = {cp['T_prior']:.3f}; both fit on DS1 out-of-fold logits)", "",
          "| Variant | DS1 OOF ECE / NLL / Brier | " + S + " single models ECE / NLL / Brier (mean) | " + S + " ensemble ECE / NLL / Brier | " + S + " ensemble macro-F1 | " + S + " ensemble S Se / PPV | " + S + " ensemble F Se / PPV |",
          "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        o = res["oof_calibration_inter"][v]
        sm = res["runs"]["inter"]["single"][v]["mean"]
        en = res["runs"]["inter"]["ensemble"][v]
        L.append(f"| {VARIANT_LABELS[v]} | {_fmt(o['ece'])} / {_fmt(o['nll'])} / {_fmt(o['brier'])} | "
                 f"{_fmt(sm['ece'])} / {_fmt(sm['nll'])} / {_fmt(sm['brier'])} | "
                 f"{_fmt(en['ece'])} / {_fmt(en['nll'])} / {_fmt(en['brier'])} | {_fmt(en['macro_f1'])} | "
                 f"{_fmt(en['per_class']['S']['sensitivity'])} / {_fmt(en['per_class']['S']['ppv'])} | "
                 f"{_fmt(en['per_class']['F']['sensitivity'])} / {_fmt(en['per_class']['F']['ppv'])} |")
    nf = res["runs"]["inter"]["bootstrap_ensemble_raw"]["resamples_without_F"]
    L += ["", f"Bootstrap resamples with no F beats: {nf} of {res['runs']['inter']['bootstrap_ensemble_raw']['n_boot']} "
          "(macro-F1 in those averages N, S, V only)."]
    return "\n".join(L) + "\n"


def _plot_confusion(cm, classes, path, title) -> None:
    import matplotlib.pyplot as plt

    from ecg import plotting as P
    P.setup()
    cm = np.array(cm, float)
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{100 * rown[i, j]:.1f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if rown[i, j] > 0.55 else P.INK)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def _plot_reliability(oof: dict, ev: dict, n_bins: int, path, split_name: str) -> None:
    import matplotlib.pyplot as plt

    from ecg import plotting as P
    P.setup()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), sharey=True)
    series = [("DS1 out-of-fold (single models)", P.INK_2, "o"),
              (f"{split_name} single models (pooled)", "#008300", "s"),
              (f"{split_name} ensemble", "#6250d6", "D")]
    for ax, v in zip(axes, VARIANTS):
        ax.plot([0, 1], [0, 1], color=P.GRID, lw=1.2, ls="--", zorder=0)
        data = [(oof["probs"][v], oof["y"]),
                (ev["probs"][v]["single_pooled"], np.tile(ev["y"], len(ev["probs"][v]["single_pooled"]) // len(ev["y"]))),
                (ev["probs"][v]["ensemble"], ev["y"])]
        eces = []
        for (lab, col, mk), (pr, yy) in zip(series, data):
            b = [r for r in reliability_bins(pr, yy, n_bins) if r[2] >= 20]  # hide bins with < 20 beats
            ax.plot([r[3] for r in b], [r[4] for r in b], color=col, marker=mk, ms=4, lw=1.6, label=lab)
            eces.append(ece(pr, yy, n_bins))
        ax.text(0.98, 0.04, "ECE\n" + "\n".join(f"{short}  {e:.3f}" for short, e in
                zip(["DS1 OOF", f"{split_name} single", f"{split_name} ensemble"], eces)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=P.INK_2,
                family="monospace")
        ax.set_title(VARIANT_LABELS[v])
        ax.set_xlabel("confidence (top-label probability)")
        ax.set_xlim(0.25, 1.01)
        ax.set_ylim(0.0, 1.01)
    axes[0].set_ylabel("accuracy")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Reliability, primary run (bins with ≥ 20 beats; dashed = perfect calibration)", y=1.02)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    import argparse

    from ecg.config import ROOT, load_config
    ap = argparse.ArgumentParser(description="Evaluate the frozen models on DS2 (once).")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="run the identical pipeline on DS1 instead")
    ap.add_argument("--overwrite", action="store_true", help="reproduce an existing DS2 result")
    a = ap.parse_args()
    evaluate_test(load_config(a.config), dry_run=a.dry_run, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
