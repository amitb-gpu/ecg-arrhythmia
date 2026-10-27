"""Summary figure and DS2 error analysis, from saved outputs only.

Reads committed metrics and the saved DS2 logits (runs/<exp>/ds2/); never
re-runs a model on DS2.

Writes
  results/summary/protocol_comparison.png   macro-F1 and per-class F1 under four protocols
  results/<exp>/ds2/error_analysis.md       where DS2 errors come from, by record
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg import evaluate as E  # noqa: E402
from ecg import plotting as P  # noqa: E402
from ecg.config import load_config, results_dir, run_dir  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text())


def comparison_figure(cfg, out):
    import matplotlib.pyplot as plt

    classes = cfg["data"]["classes"]
    lit = load(results_dir(cfg, "intra_all44") / "test_metrics.json")["test"]
    beat = load(results_dir(cfg, "intra_ds1") / "oof_metrics.json")["pooled_oof"]
    pat = load(results_dir(cfg, "inter") / "oof_metrics.json")["pooled_oof"]
    ds2 = load(results_dir(cfg, "ds2") / "ds2_metrics.json")["runs"]["inter"]["ensemble"]["raw"]
    settings = [("Random beat split,\nall 44 records\n(literature protocol)", lit),
                ("DS1 CV,\nbeat-level folds", beat),
                ("DS1 CV,\npatient-level folds", pat),
                ("DS2: 22 unseen\npatients (primary)", ds2)]
    P.setup()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    series = [("macro-F1", P.INK_2)] + [(f"{c} F1", P.CLASS_COLORS[c]) for c in classes]
    w = 0.15
    for j, (lab, col) in enumerate(series):
        xs = np.arange(len(settings)) + (j - 2) * w
        vals = [m["macro_f1"] if j == 0 else m["per_class"][classes[j - 1]]["f1"] for _, m in settings]
        ax.bar(xs, vals, w * 0.88, color=col, label=lab, zorder=2)
        if j == 0:
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                        color=P.INK, fontweight="bold")
    ax.axvline(1.5, color=P.INK_2, lw=0.8, ls=":")
    ax.text(0.5, 1.13, "same patients in train and test", ha="center", fontsize=8.5, color=P.INK_2)
    ax.text(2.5, 1.13, "test patients never seen in training", ha="center", fontsize=8.5, color=P.INK_2)
    ax.set_xticks(range(len(settings)), [s for s, _ in settings], fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("F1")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5)
    ax.set_title("Same model and config, four evaluation protocols", pad=10)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "protocol_comparison.png")
    plt.close(fig)


def error_analysis(cfg):
    classes = cfg["data"]["classes"]
    L = np.load(run_dir(cfg, "ds2") / "logits_inter.npz")
    y, rec = L["y"], L["record"]
    pred = softmax(L["logits"], axis=2).mean(0).argmax(1)
    s, f, n = classes.index("S"), classes.index("F"), classes.index("N")

    def se_ppv(mask, c):
        tp = ((pred == c) & (y == c) & mask).sum()
        return tp / max(((y == c) & mask).sum(), 1), tp / max(((pred == c) & mask).sum(), 1)

    all_ = np.ones_like(y, bool)
    lines = ["## DS2 error analysis (primary ensemble, raw argmax; from saved predictions)", ""]
    s_by = {int(r): int(((y == s) & (rec == r)).sum()) for r in np.unique(rec)}
    top = max(s_by, key=s_by.get)
    m232 = rec == top
    lines += [f"**S is dominated by record {top}:** {s_by[top]:,} of {(y == s).sum():,} S beats "
              f"({100 * s_by[top] / (y == s).sum():.0f}%).", "",
              "| S beats | n | Se | PPV |", "|---|---|---|---|"]
    for lab, mk in ((f"all DS2", all_), (f"record {top} only", m232), (f"DS2 without {top}", ~m232)):
        se, ppv = se_ppv(mk, s)
        lines.append(f"| {lab} | {int(((y == s) & mk).sum()):,} | {se:.3f} | {ppv:.3f} |")
    where = {classes[k]: int(((y == s) & m232 & (pred == k)).sum()) for k in range(len(classes))}
    lines += ["", f"Record {top}'s S beats were predicted as: " +
              ", ".join(f"{k} {v:,}" for k, v in where.items()) + ".", ""]

    f_by = {int(r): int(((y == f) & (rec == r)).sum()) for r in np.unique(rec)}
    topf = max(f_by, key=f_by.get)
    pf = {classes[k]: int(((y == f) & (rec == topf) & (pred == k)).sum()) for k in range(len(classes))}
    lines += [f"**F is dominated by record {topf}:** {f_by[topf]} of {(y == f).sum()} F beats; "
              f"predicted as " + ", ".join(f"{k} {v}" for k, v in pf.items()) + ".", ""]

    nf = {int(r): int(((y == n) & (pred == f) & (rec == r)).sum()) for r in np.unique(rec)}
    tot = sum(nf.values())
    lines += [f"**N beats called F: {tot:,}.** Largest sources: " +
              ", ".join(f"{r} ({v})" for r, v in sorted(nf.items(), key=lambda kv: -kv[1])[:4]) + ".", ""]
    nv = {int(r): int(((y == n) & (pred == classes.index("V")) & (rec == r)).sum()) for r in np.unique(rec)}
    lines += [f"**N beats called V: {sum(nv.values()):,}.** Largest sources: " +
              ", ".join(f"{r} ({v})" for r, v in sorted(nv.items(), key=lambda kv: -kv[1])[:4]) + ".", ""]
    # Secondary vs primary, paired over the same resampled records (analysis only;
    # the primary was fixed before DS2 and does not change).
    L2 = np.load(run_dir(cfg, "ds2") / "logits_inter_fixed12.npz")
    pred2 = softmax(L2["logits"], axis=2).mean(0).argmax(1)
    recs = np.unique(rec)
    cm1 = np.stack([E.confusion(y[rec == r], pred[rec == r], len(classes)) for r in recs])
    cm2 = np.stack([E.confusion(y[rec == r], pred2[rec == r], len(classes)) for r in recs])
    draws = np.random.default_rng(cfg["seed"]).integers(0, len(recs), (2000, len(recs)))
    with np.errstate(invalid="ignore"):
        diff = np.array([E.metrics_from_cm(cm2[i].sum(0), classes)["macro_f1"] -
                         E.metrics_from_cm(cm1[i].sum(0), classes)["macro_f1"] for i in draws])
    point = E.class_metrics(y, pred2, classes)["macro_f1"] - E.class_metrics(y, pred, classes)["macro_f1"]
    s2 = ((pred2 == s) & (y == s) & m232).sum() / (m232 & (y == s)).sum()
    s2o = ((pred2 == s) & (y == s) & ~m232).sum() / (~m232 & (y == s)).sum()
    lines += ["**Secondary (fixed 12 epochs) vs primary (early-stopped), paired record bootstrap:** "
              f"macro-F1 difference {point:+.3f}, 95% CI [{np.percentile(diff, 2.5):+.3f}, "
              f"{np.percentile(diff, 97.5):+.3f}], secondary higher in {100 * (diff > 0).mean():.0f}% of resamples. "
              f"S Se on record {top}: {s2:.3f} (primary {se_ppv(m232, s)[0]:.3f}); "
              f"on the other records: {s2o:.3f} (primary {se_ppv(~m232, s)[0]:.3f}). "
              "Reported for comparison only; the primary result was fixed before DS2 was opened.", ""]
    text = "\n".join(lines) + "\n"
    (results_dir(cfg, "ds2") / "error_analysis.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = load_config(ap.parse_args().config)
    comparison_figure(cfg, ROOT / cfg["paths"]["results_dir"] / "summary")
    error_analysis(cfg)


if __name__ == "__main__":
    main()
