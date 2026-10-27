"""Build cache/ds1.npz, cache/ds2.npz and cache/folds.json, then report
class counts per fold and split.

DS2 is processed with the exact preprocessing fixed on DS1 (all per-record);
only its class totals are reported here, for the record.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg import data as D  # noqa: E402
from ecg.config import load_config  # noqa: E402


def counts(y: np.ndarray) -> dict:
    return {c: int((y == i).sum()) for i, c in enumerate(D.AAMI_CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = load_config(ap.parse_args().config)

    info = D.build_dataset(cfg, ROOT)
    d1 = D.load_split(cfg, ROOT, "ds1", drop_q=False)
    d2 = D.load_split(cfg, ROOT, "ds2", drop_q=False)
    folds = info["folds"]

    rows = []
    for k in range(cfg["cv"]["n_folds"]):
        recs = sorted(r for r, f in folds.items() if f == k)
        m = np.isin(d1["record"], recs)
        rows.append({"split": f"DS1 fold {k}", "records": " ".join(map(str, recs)),
                     **counts(d1["y"][m])})
    rows.append({"split": "DS1 total", "records": "22 records", **counts(d1["y"])})
    rows.append({"split": "DS2 total", "records": "22 records", **counts(d2["y"])})
    df = pd.DataFrame(rows).set_index("split")
    df["total (excl. Q)"] = df[["N", "S", "V", "F"]].sum(1)

    out = ROOT / cfg["paths"]["results_dir"] / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "split_fold_counts.csv")
    (out / "build_info.json").write_text(json.dumps(
        {k: v for k, v in info.items() if k != "folds"}, indent=1))

    pd.set_option("display.width", 200)
    print(df.to_string())
    print(f"\nDropped beats: DS1 {info['ds1']['n_drop_edge']} edge + "
          f"{info['ds1']['n_drop_no_rr']} no-RR; DS2 {info['ds2']['n_drop_edge']} edge + "
          f"{info['ds2']['n_drop_no_rr']} no-RR")
    print(f"Arrays: x {d1['x'].shape} {d1['x'].dtype}, rr {d1['rr'].shape}, "
          f"RR features {D.RR_FEATURES}")


if __name__ == "__main__":
    main()
