"""Phase 2 inspection of DS1 only: per-record counts, drop accounting,
validation-strategy comparison, and a sanity plot of beat windows.

Reads no DS2 data. Outputs go to results/phase2/.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg import data as D  # noqa: E402
from ecg import plotting as P  # noqa: E402

OUT = ROOT / "results" / "phase2"
SEED = 0
K = 5
# Published DS1 totals (Huang et al. 2014, Table 1), before any beat drops.
PUBLISHED_DS1 = {"N": 45868, "S": 942, "V": 3787, "F": 415, "Q": 8}


def per_record_table(mitdb: Path) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    rows, beats, other = [], [], Counter()
    for rec in D.DS1:
        r = D.load_annotations(rec, mitdb)
        m = D.beat_keep_mask(r)
        lab = m["labels"]
        row = {"record": rec}
        for c in D.AAMI_CLASSES:
            row[c] = int(((lab == c) & m["keep"]).sum())
        row["kept"] = int(m["keep"].sum())
        row["all_aami"] = int(m["in_aami"].sum())
        row["drop_edge"] = int((m["in_aami"] & m["edge"]).sum())
        row["drop_no_rr"] = int((m["in_aami"] & m["no_rr"] & ~m["edge"]).sum())
        row["qrs_not_aami"] = int((~m["in_aami"]).sum())
        other.update(r.beat_symbols[~m["in_aami"]].tolist())
        rows.append(row)
        for c in lab[m["keep"]]:
            beats.append((rec, c))
    df = pd.DataFrame(rows).set_index("record")
    bdf = pd.DataFrame(beats, columns=["record", "label"])
    return df, bdf, other


def fold_counts(bdf: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for i, (_, va) in enumerate(folds):
        v = bdf.iloc[va]
        recs = sorted(v.record.unique().tolist())
        row = {"fold": i, "records": " ".join(map(str, recs))}
        row.update({c: int((v.label == c).sum()) for c in D.AAMI_CLASSES})
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def sanity_plot(mitdb: Path, bdf_records: list[int], path: Path) -> None:
    """A few beats per class from DS1, filtered + normalized, R-peak marked."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    pool = {c: [] for c in D.AAMI_CLASSES}
    for rec in bdf_records:
        r = D.load_record(rec, mitdb)
        m = D.beat_keep_mask(r)
        xf = D.bandpass(r.signal)
        for c in D.AAMI_CLASSES:
            idx = np.where(m["keep"] & (m["labels"] == c))[0]
            if len(idx):
                pick = rng.choice(idx, size=min(4, len(idx)), replace=False)
                w = D.normalize_beats(D.extract_windows(xf, r.beat_samples[pick]))
                pool[c].extend((rec, wi) for wi in w)
    P.setup()
    t_ms = (np.arange(D.WIN) - D.PRE) / D.FS * 1000
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for ax, c in zip(axes, D.AAMI_CLASSES):
        items = pool[c]
        sel = rng.choice(len(items), size=min(8, len(items)), replace=False)
        recs = sorted({items[i][0] for i in sel})
        for i in sel:
            ax.plot(t_ms, items[i][1], color=P.CLASS_COLORS[c], alpha=0.55, lw=1.0)
        ax.axvline(0, color=P.INK_2, lw=0.8, ls="--")
        ax.set_title(f"{c}  (n={len(sel)} beats, {len(recs)} records)")
        ax.set_xlabel("ms from annotated R-peak")
    axes[0].set_ylabel("z-scored amplitude")
    fig.suptitle("DS1 example beats per AAMI class: 0.5-40 Hz bandpass, "
                 "-250/+450 ms window, per-beat z-score", y=1.04)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mitdb = D.download(ROOT / "data")
    D.check_split(mitdb)
    print("Split check: DS1, DS2, paced are disjoint and cover all 48 records.")

    # MLII present in every DS1 record?
    import wfdb
    no_mlii = [r for r in D.DS1
               if "MLII" not in wfdb.rdheader(str(mitdb / str(r))).sig_name]
    print(f"DS1 records without MLII: {no_mlii or 'none'}")

    df, bdf, other = per_record_table(mitdb)
    tot = df.sum()
    df.loc["TOTAL"] = tot
    df.to_csv(OUT / "ds1_per_record_counts.csv")
    pd.set_option("display.width", 200)
    print("\nDS1 per-record counts (kept beats per class, plus drops):")
    print(df.to_string())
    print(f"\nQRS annotations outside the AAMI mapping (kept for RR only): {dict(other)}")

    # Cross-check against published totals (before drops).
    pre = {c: int(tot[c]) for c in D.AAMI_CLASSES}
    dropped_by_class = Counter()
    for rec in D.DS1:
        r = D.load_annotations(rec, mitdb)
        m = D.beat_keep_mask(r)
        dropped_by_class.update(m["labels"][m["in_aami"] & ~m["keep"]].tolist())
    print("\nClass | kept | dropped | kept+dropped | published (Huang 2014)")
    for c in D.AAMI_CLASSES:
        print(f"  {c}   | {pre[c]:5d} | {dropped_by_class[c]:3d} | "
              f"{pre[c] + dropped_by_class[c]:5d} | {PUBLISHED_DS1[c]}")

    # Share of each minority class held by the single largest record.
    print("\nConcentration (share of class held by top records):")
    for c in ["S", "V", "F"]:
        s = df.drop("TOTAL")[c].sort_values(ascending=False)
        top = ", ".join(f"{r}:{v}" for r, v in s.head(4).items() if v)
        print(f"  {c}: top record {s.iloc[0] / s.sum():.0%} | {top}")

    # Validation strategies.
    y, g = bdf.label.values, bdf.record.values
    res = {}
    res["GroupKFold"] = fold_counts(bdf, GroupKFold(K).split(bdf, y, g))
    for seed in range(3):
        sgkf = StratifiedGroupKFold(K, shuffle=True, random_state=seed)
        res[f"StratifiedGroupKFold(seed={seed})"] = fold_counts(bdf, sgkf.split(bdf, y, g))
    with open(OUT / "cv_fold_counts.txt", "w") as f:
        for name, fc in res.items():
            block = f"\n{name} (k={K}) - validation-fold class counts:\n{fc.to_string()}\n"
            print(block)
            f.write(block)

    sanity_plot(mitdb, D.DS1, ROOT / "results" / "ds1_example_beats.png")
    print("Saved results/ds1_example_beats.png")


if __name__ == "__main__":
    main()
