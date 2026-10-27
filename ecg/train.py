"""Cross-validated training on DS1. DS2 is never loaded here.

Schemes
  inter      StratifiedGroupKFold over DS1 *records* (patient-level).
  intra_ds1  StratifiedKFold over DS1 *beats* (naive; same patient's beats end
             up in both train and validation). Same model, same data, same
             budget: the only difference is how beats are assigned to folds.

Per fold: fit RR scaler and class weights on the training records only, train
with early stopping on class-weighted validation loss (or a fixed number of
epochs), save the best checkpoint, and store out-of-fold (OOF) logits.

Usage
  python -m ecg.train --scheme inter
  python -m ecg.train --scheme inter --fixed-epochs-from inter
  python -m ecg.train --scheme intra_ds1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold

from ecg import data as D
from ecg import evaluate as E
from ecg.config import ROOT, load_config, run_dir as get_run_dir
from ecg.model import build_model


# ---------------------------------------------------------------- runtime

def setup_runtime(cfg: dict) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("No GPU visible to TensorFlow; refusing to train on CPU. "
                         "Run scripts/check_gpu.py.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    if cfg["train"]["deterministic"]:
        tf.config.experimental.enable_op_determinism()
    if cfg["train"]["mixed_precision"]:
        keras.mixed_precision.set_global_policy("mixed_float16")


def class_weights(y: np.ndarray, n_classes: int, mode: str = "balanced") -> np.ndarray:
    """Per-class loss weights, computed from the training fold only.

    balanced:      w_c = n / (K n_c); every class contributes equally to the
                   loss in expectation. Explodes when a class is nearly absent
                   (Run 1, fold 0: F weighed ~1,000x N).
    sqrt_balanced: w_c proportional to sqrt(n / (K n_c)), rescaled so the mean
                   per-sample weight is 1 (same loss scale as unweighted).
    """
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(float), 1)
    w = len(y) / (n_classes * counts)
    if mode == "balanced":
        return w
    if mode == "sqrt_balanced":
        w = np.sqrt(w)
        return w * len(y) / (w * counts).sum()
    raise ValueError(mode)


def make_ds(x, rr, y, w, batch: int, shuffle: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices(((x[..., None], rr), y, w))
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


class ValMacroF1(keras.callbacks.Callback):
    """Logs validation macro-F1 (all classes, and N/S/V only) each epoch.
    Must run before the checkpoint/early-stopping callbacks that read it."""

    def __init__(self, x, rr, y, n_classes):
        super().__init__()
        self.inputs, self.y, self.k = (x[..., None], rr), y, n_classes

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.inputs, batch_size=4096, verbose=0).argmax(1)
        m = E.class_metrics(self.y, pred, [str(i) for i in range(self.k)])
        logs["val_macro_f1"] = m["macro_f1"]
        # N, S, V (codes 0-2) are present in every patient-level fold; F is not.
        logs["val_macro_f1_nsv"] = float(np.mean([m["per_class"][str(i)]["f1"] for i in range(3)]))


# ---------------------------------------------------------------- one fold

def train_fold(cfg: dict, d: dict, tr: np.ndarray, va: np.ndarray, fold: int,
               fold_dir: Path, fixed_epochs: int | None) -> dict:
    tcfg, n_cls = cfg["train"], len(cfg["data"]["classes"])
    seed = cfg["seed"] + fold
    keras.utils.set_random_seed(seed)

    scaler = D.RRScaler().fit(d["rr"][tr])          # training records only
    rr_tr, rr_va = scaler.transform(d["rr"][tr]), scaler.transform(d["rr"][va])
    cw = class_weights(d["y"][tr], n_cls, tcfg["class_weight"])  # training records only
    w_tr, w_va = cw[d["y"][tr]].astype(np.float32), cw[d["y"][va]].astype(np.float32)

    ds_tr = make_ds(d["x"][tr], rr_tr, d["y"][tr], w_tr, tcfg["batch_size"], True, seed)
    ds_va = make_ds(d["x"][va], rr_va, d["y"][va], w_va, 4096, False, seed)

    model = build_model(cfg["model"], d["x"].shape[1], rr_tr.shape[1], n_cls)
    model.compile(
        optimizer=keras.optimizers.AdamW(tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[],
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = fold_dir / "best.keras"
    monitor = tcfg["early_stop_monitor"]
    mode = "min" if monitor.endswith("loss") else "max"
    cbs = [ValMacroF1(d["x"][va], rr_va, d["y"][va], n_cls)]
    if fixed_epochs is None:
        epochs = tcfg["max_epochs"]
        cbs += [keras.callbacks.ModelCheckpoint(ckpt, monitor=monitor, mode=mode, save_best_only=True),
                keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=tcfg["patience"])]
    else:
        epochs = fixed_epochs

    t0 = time.perf_counter()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=cbs, verbose=0)
    secs = time.perf_counter() - t0
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}

    if fixed_epochs is None:
        pick = np.argmin if mode == "min" else np.argmax  # first best, as the checkpoint keeps
        best_epoch = int(pick(h[monitor])) + 1
        model = keras.models.load_model(ckpt)
    else:
        best_epoch = fixed_epochs
        model.save(ckpt)

    logits = model.predict((d["x"][va][..., None], rr_va), batch_size=4096, verbose=0)
    np.savez(fold_dir / "val_logits.npz", logits=logits, idx=va)
    (fold_dir / "history.json").write_text(json.dumps(h))
    (fold_dir / "scaler.json").write_text(json.dumps(
        {"mean": scaler.mean_.tolist(), "std": scaler.std_.tolist(), "features": D.RR_FEATURES}))
    return {"fold": fold, "best_epoch": best_epoch, "epochs_run": len(h["loss"]),
            "seconds": round(secs, 1), "class_weights": cw.tolist(),
            "n_train": int(len(tr)), "n_val": int(len(va)), "logits": logits,
            "policy": keras.mixed_precision.global_policy().name,
            "optimizer": type(model.optimizer).__name__,
            "device": model.get_layer("stem_conv").kernel.value.device}


# ---------------------------------------------------------------- cross-validation

def fold_ids(cfg: dict, d: dict, scheme: str) -> np.ndarray:
    if scheme == "inter":
        folds = D.load_folds(cfg, ROOT)
        return np.array([folds[int(r)] for r in d["record"]])
    if scheme == "intra_ds1":
        skf = StratifiedKFold(cfg["cv"]["n_folds"], shuffle=True, random_state=cfg["seed"])
        ids = np.empty(len(d["y"]), int)
        for k, (_, va) in enumerate(skf.split(d["y"], d["y"])):
            ids[va] = k
        return ids
    raise ValueError(scheme)


def run_cv(cfg: dict, scheme: str, run_name: str, fixed_epochs: int | None) -> dict:
    setup_runtime(cfg)
    d = D.load_split(cfg, ROOT, "ds1", drop_q=True)
    ids = fold_ids(cfg, d, scheme)
    run_dir = get_run_dir(cfg, run_name)
    oof = np.full((len(d["y"]), len(cfg["data"]["classes"])), np.nan, np.float32)

    folds = []
    for k in range(cfg["cv"]["n_folds"]):
        tr, va = np.where(ids != k)[0], np.where(ids == k)[0]
        if scheme == "inter":  # patient-level: no record on both sides
            assert not set(d["record"][tr]) & set(d["record"][va])
        r = train_fold(cfg, d, tr, va, k, run_dir / f"fold{k}", fixed_epochs)
        oof[va] = r.pop("logits")
        folds.append(r)
        print(f"[{run_name}] fold {k}: best epoch {r['best_epoch']}/{r['epochs_run']}, "
              f"{r['seconds']}s, {r['policy']}, {r['optimizer']}, {r['device']}", flush=True)

    assert np.isfinite(oof).all()
    np.savez(run_dir / "oof.npz", logits=oof, y=d["y"], record=d["record"],
             fold=ids, sample=d["sample"])
    summary = {"scheme": scheme, "run_name": run_name, "fixed_epochs": fixed_epochs,
               "folds": folds,
               "median_best_epoch": int(np.median([f["best_epoch"] for f in folds]))}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--scheme", choices=["inter", "intra_ds1"], default="inter")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--fixed-epochs-from", default=None,
                    help="run name whose median best epoch to train for, without early stopping")
    a = ap.parse_args()
    cfg = load_config(a.config)

    fixed = None
    if a.fixed_epochs_from:
        src = json.loads((get_run_dir(cfg, a.fixed_epochs_from) / "summary.json").read_text())
        fixed = src["median_best_epoch"]
    name = a.run_name or (a.scheme + (f"_fixed{fixed}" if fixed else ""))
    run_cv(cfg, a.scheme, name, fixed)
    from ecg.evaluate import report_cv
    report_cv(cfg, name)


if __name__ == "__main__":
    main()
