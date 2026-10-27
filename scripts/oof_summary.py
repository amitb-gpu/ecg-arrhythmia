"""Out-of-fold summary table for one experiment (DS1 only).

Writes results/<experiment>/oof_summary.md and prints it.
Usage: python scripts/oof_summary.py --config configs/default.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg.config import load_config, results_dir  # noqa: E402


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = load_config(ap.parse_args().config)
    exp_dir = ROOT / cfg["paths"]["results_dir"] / cfg["experiment"]
    runs = sorted(p.name for p in exp_dir.iterdir() if (p / "oof_metrics.json").exists())
    order = [r for r in runs if r == "inter"] + [r for r in runs if r.startswith("inter_fixed")] + \
            [r for r in runs if r.startswith("intra")]
    labels = {"inter": "Patient-level CV, early stop", "intra_ds1": "Beat-level CV (leaky)"}

    ms = {r: json.loads((results_dir(cfg, r) / "oof_metrics.json").read_text()) for r in order}
    pf = {r: pd.read_csv(results_dir(cfg, r) / "per_fold_metrics.csv") for r in order}
    head = [labels.get(r, f"Patient-level CV, fixed {r.removeprefix('inter_fixed')} epochs") for r in order]

    lines = [f"## {cfg['experiment']}: out-of-fold results on DS1 (DS2 untouched)", "",
             "| | " + " | ".join(head) + " |", "|---" * (len(order) + 1) + "|"]
    p = {r: ms[r]["pooled_oof"] for r in order}
    lines.append("| macro-F1 (N,S,V,F) | " + " | ".join(fmt(p[r]["macro_f1"]) for r in order) + " |")
    for c in "NSVF":
        lines.append(f"| {c} Se / PPV | " + " | ".join(
            f"{fmt(p[r]['per_class'][c]['sensitivity'])} / {fmt(p[r]['per_class'][c]['ppv'])}" for r in order) + " |")
    lines.append("| per-fold macro-F1 (N,S,V) | " + " | ".join(
        f"{ms[r]['per_fold_macroF1_NSV_mean']:.3f} ± {ms[r]['per_fold_macroF1_NSV_std']:.3f}" for r in order) + " |")
    lines.append("| accuracy | " + " | ".join(fmt(p[r]["accuracy"]) for r in order) + " |")
    lines.append("| ECE / NLL (raw) | " + " | ".join(
        f"{fmt(p[r]['ece_uncalibrated'])} / {fmt(p[r]['nll_uncalibrated'])}" for r in order) + " |")
    lines.append("| best epochs | " + " | ".join(", ".join(map(str, pf[r]["best_epoch"])) for r in order) + " |")

    if "inter" in pf:
        f = pf["inter"]
        lines += ["", "### Patient-level CV per fold (F shown only for the fold holding record 208)", "",
                  "| fold | records | best epoch | N Se / PPV | S n | S Se / PPV | V n | V Se / PPV | F n | F Se / PPV | macro-F1 (N,S,V) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, row in f.iterrows():
            lines.append(
                f"| {row.fold} | {row.records} | {row.best_epoch} | {fmt(row.N_Se)} / {fmt(row.N_PPV)} | "
                f"{row.S_n} | {fmt(row.S_Se)} / {fmt(row.S_PPV)} | {row.V_n} | {fmt(row.V_Se)} / {fmt(row.V_PPV)} | "
                f"{row.F_n} | {fmt(row.F_Se)} / {fmt(row.F_PPV)} | {fmt(row.macroF1_NSV)} |")
        cm = ms["inter"]["pooled_oof"]["confusion"]
        lines += ["", "Patient-level pooled confusion matrix (rows = true, columns = predicted):", "",
                  "| | N | S | V | F |", "|---|---|---|---|---|"]
        lines += [f"| {c} | " + " | ".join(f"{v:,}" for v in cm[i]) + " |" for i, c in enumerate("NSVF")]

    text = "\n".join(lines) + "\n"
    (exp_dir / "oof_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
