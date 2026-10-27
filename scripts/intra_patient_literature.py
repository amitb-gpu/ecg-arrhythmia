"""Literature-style intra-patient evaluation, for comparison only.

Many published MIT-BIH results pool the beats of all 44 non-paced records and
split them at random, so every test patient also contributes training beats.
This script reproduces that protocol with the frozen model and config:

  - all 44 records (DS1 + DS2), Q excluded
  - stratified random beat split: 70% train, 10% validation, 20% test
  - one model, same config and training code as the patient-level runs

It trains on DS2 beats, so it runs only after the DS2 patient-level result
was committed (docs/decisions.md). Nothing here feeds back into any choice.

Usage: python scripts/intra_patient_literature.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg import data as D  # noqa: E402
from ecg import evaluate as E  # noqa: E402
from ecg.config import load_config, results_dir, run_dir  # noqa: E402

RUN = "intra_all44"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = load_config(ap.parse_args().config)
    if not (results_dir(cfg, "ds2") / "ds2_metrics.json").exists():
        raise SystemExit("Run the patient-level DS2 evaluation first (python -m ecg.evaluate).")

    import keras
    from ecg.train import setup_runtime, train_fold
    setup_runtime(cfg)

    d1, d2 = (D.load_split(cfg, ROOT, s, drop_q=True) for s in ("ds1", "ds2"))
    d = {k: np.concatenate([d1[k], d2[k]]) for k in ("x", "rr", "y", "record", "sample")}
    idx = np.arange(len(d["y"]))
    trva, te = train_test_split(idx, test_size=0.2, stratify=d["y"], random_state=cfg["seed"])
    tr, va = train_test_split(trva, test_size=0.125, stratify=d["y"][trva], random_state=cfg["seed"])

    fold_dir = run_dir(cfg, RUN) / "model"
    r = train_fold(cfg, d, tr, va, 0, fold_dir, fixed_epochs=None)
    r.pop("logits")

    sc = json.loads((fold_dir / "scaler.json").read_text())
    rr = ((np.log(d["rr"][te]) - np.array(sc["mean"])) / np.array(sc["std"])).astype(np.float32)
    model = keras.models.load_model(fold_dir / "best.keras", compile=False)
    logits = model.predict((d["x"][te][..., None], rr), batch_size=4096, verbose=0).astype(np.float64)

    classes = cfg["data"]["classes"]
    m = E.class_metrics(d["y"][te], logits.argmax(1), classes)
    test_recs, train_recs = set(d["record"][te]), set(d["record"][tr])
    out = results_dir(cfg, RUN)
    out.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "random stratified beat split over 44 records: 70/10/20",
              "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
              "test_records_also_in_training": f"{len(test_recs & train_recs)} of {len(test_recs)}",
              "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "seconds": r["seconds"],
              "test": m}
    (out / "test_metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "test"}, indent=1))
    print(f"test macro-F1 {m['macro_f1']:.3f}")
    for c in classes:
        pc = m["per_class"][c]
        print(f"  {c}: Se {pc['sensitivity']:.3f}  PPV {pc['ppv']:.3f}  F1 {pc['f1']:.3f}  (n={pc['support']})")


if __name__ == "__main__":
    main()
