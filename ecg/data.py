"""MIT-BIH loading, AAMI mapping, filtering, beat windows, RR features,
dataset build and fold assignment.

Everything that touches signals is per-record: filtering, windowing,
per-beat normalization and RR features use only that patient's data. The one
statistic fit across beats, RR-feature standardization (`RRScaler`), is fit on
training records only, inside each fold.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold

FS = 360  # Hz, MIT-BIH sampling rate

# de Chazal et al. (2004) inter-patient split. Verified against
# Guo, Sim & Matuszewski 2019 (arXiv:1810.04121), which prints both lists, and
# Mousavi et al. 2019 (arXiv:1812.07421) for DS2. Other papers contain
# transcription errors (duplicated 101/220, "7"/"11" for 113/117).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
PACED = [102, 104, 107, 217]  # excluded: paced rhythms

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]  # label codes 0..4 in built files
AAMI = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
    **dict.fromkeys(["A", "a", "J", "S"], "S"),
    **dict.fromkeys(["V", "E"], "V"),
    "F": "F",
    **dict.fromkeys(["/", "f", "Q"], "Q"),
}
Q_CODE = AAMI_CLASSES.index("Q")
# Symbols that mark a QRS complex. Used for RR intervals, so a beat we don't
# classify (e.g. Q) still counts as the neighbour of the beats around it.
# '!' (ventricular flutter wave) and non-beat annotations are not QRS marks.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Defaults; build_dataset() takes the values from the config.
PRE = int(0.25 * FS)   # 90 samples before the R-peak
POST = int(0.45 * FS)  # 162 samples after
WIN = PRE + POST       # 252 samples; R-peak at index PRE
LOCAL_RR_BEATS = 10

RR_FEATURES = ["pre_rr", "post_rr", "local_rr", "pre_over_local", "post_over_pre"]

MIRROR = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/"
CHECKSUMS = "https://physionet.org/files/mitdb/1.0.0/SHA256SUMS.txt"


# ---------------------------------------------------------------- download

def download(data_dir: Path) -> Path:
    """Fetch .hea/.dat/.atr for all 48 records and verify SHA-256.

    Files come from PhysioNet's open-data mirror on AWS (physionet.org itself
    is very slow to serve 150 files). Checksums come from physionet.org, so a
    corrupted or altered mirror file fails loudly.
    """
    out = Path(data_dir) / "mitdb"
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / "SHA256SUMS.txt"
    if not sums_path.exists():
        urllib.request.urlretrieve(CHECKSUMS, sums_path)
    expected = dict(line.split()[::-1] for line in sums_path.read_text().splitlines() if line.strip())

    for rec in sorted(DS1 + DS2 + PACED):
        for ext in ("hea", "dat", "atr"):
            name = f"{rec}.{ext}"
            path = out / name
            if not path.exists():
                urllib.request.urlretrieve(MIRROR + name, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected[name]:
                path.unlink()
                raise RuntimeError(f"checksum mismatch for {name}")
    return out


def check_split(mitdb_dir: Path) -> None:
    """DS1, DS2 and paced records are disjoint and cover all 48 records
    (record list taken from PhysioNet's checksum file)."""
    s1, s2, sp = set(DS1), set(DS2), set(PACED)
    assert len(s1) == len(DS1) == 22 and len(s2) == len(DS2) == 22
    assert not (s1 & s2) and not (s1 & sp) and not (s2 & sp)
    sums = (Path(mitdb_dir) / "SHA256SUMS.txt").read_text().split()
    all48 = {int(n[:-4]) for n in sums if n.endswith(".dat") and n[:-4].isdigit()}
    assert len(all48) == 48 and s1 | s2 | sp == all48


# ---------------------------------------------------------------- records

@dataclass
class Record:
    name: int
    signal: np.ndarray        # chosen lead, mV, unfiltered (empty if not loaded)
    beat_samples: np.ndarray  # sample index of every QRS annotation
    beat_symbols: np.ndarray  # original MIT-BIH symbol per QRS annotation
    n_samples: int


def _qrs_annotations(path: str) -> tuple[np.ndarray, np.ndarray]:
    ann = wfdb.rdann(path, "atr")
    sym = np.array(ann.symbol)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    return ann.sample[keep], sym[keep]


def load_record(rec: int, mitdb_dir: Path, lead: str = "MLII") -> Record:
    path = str(Path(mitdb_dir) / str(rec))
    sig, fields = wfdb.rdsamp(path)
    names = fields["sig_name"]
    if lead not in names:
        raise ValueError(f"record {rec} has no {lead} lead: {names}")
    x = sig[:, names.index(lead)].astype(np.float64)
    s, sym = _qrs_annotations(path)
    return Record(rec, x, s, sym, len(x))


def load_annotations(rec: int, mitdb_dir: Path) -> Record:
    """Annotations and record length only (no signal), for fast counting."""
    path = str(Path(mitdb_dir) / str(rec))
    s, sym = _qrs_annotations(path)
    return Record(rec, np.empty(0), s, sym, wfdb.rdheader(path).sig_len)


# ---------------------------------------------------------------- per-record processing

def bandpass(x: np.ndarray, lo: float = 0.5, hi: float = 40.0,
             order: int = 2, fs: int = FS) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    0.5 Hz removes baseline wander (breathing, movement); 40 Hz removes
    powerline (60 Hz in the US recordings) and EMG noise while keeping QRS
    energy (mostly 5-25 Hz). Zero-phase (forward-backward) so the R-peak
    doesn't shift relative to the annotation. This is offline/non-causal; a
    streaming wearable would need a causal filter and a fixed delay.
    """
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def rr_features(beat_samples: np.ndarray, local_beats: int = LOCAL_RR_BEATS,
                fs: int = FS) -> np.ndarray:
    """Per-beat RR features (seconds / ratios), columns as RR_FEATURES.

    pre/post use neighbouring QRS annotations of any type. Local average is
    the mean of up to the preceding `local_beats` intervals (including pre).
    The ratios express prematurity relative to this patient's own rhythm.
    First beat has no pre-RR and last beat no post-RR: those rows are NaN and
    the beats are dropped.
    """
    t = beat_samples / fs
    rr = np.diff(t)
    n = len(t)
    pre = np.full(n, np.nan)
    post = np.full(n, np.nan)
    pre[1:] = rr
    post[:-1] = rr
    csum = np.concatenate([[0.0], np.cumsum(rr)])
    local = np.full(n, np.nan)
    for i in range(1, n):
        j0 = max(0, i - local_beats)
        local[i] = (csum[i] - csum[j0]) / (i - j0)
    return np.stack([pre, post, local, pre / local, post / pre], axis=1)


def aami_labels(symbols: np.ndarray) -> np.ndarray:
    """AAMI class per symbol, or '' for QRS types outside the mapping."""
    return np.array([AAMI.get(s, "") for s in symbols])


def beat_keep_mask(rec: Record, pre: int = PRE, post: int = POST) -> dict[str, np.ndarray]:
    """Which annotated beats survive, and why the others are dropped."""
    s = rec.beat_samples
    labels = aami_labels(rec.beat_symbols)
    in_aami = labels != ""
    edge = (s - pre < 0) | (s + post > rec.n_samples)
    no_rr = np.zeros(len(s), bool)
    no_rr[0] = no_rr[-1] = True  # first beat: no pre-RR; last: no post-RR
    keep = in_aami & ~edge & ~no_rr
    return {"labels": labels, "in_aami": in_aami, "edge": edge,
            "no_rr": no_rr, "keep": keep}


def extract_windows(x_filt: np.ndarray, centers: np.ndarray,
                    pre: int = PRE, post: int = POST) -> np.ndarray:
    idx = centers[:, None] + np.arange(-pre, post)[None, :]
    return x_filt[idx]


def normalize_beats(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-beat z-score. Removes gain/offset differences between patients and
    electrode placements; uses only that beat, so it cannot leak."""
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return (w - mu) / (sd + eps)


# ---------------------------------------------------------------- dataset build

def process_record(rec: int, mitdb_dir: Path, dcfg: dict) -> dict[str, np.ndarray]:
    pre = int(round(dcfg["window_s"][0] * FS))
    post = int(round(dcfg["window_s"][1] * FS))
    lo, hi = dcfg["bandpass_hz"]
    r = load_record(rec, mitdb_dir, dcfg["lead"])
    m = beat_keep_mask(r, pre, post)
    keep = m["keep"]
    xf = bandpass(r.signal, lo, hi, dcfg["filter_order"])
    centers = r.beat_samples[keep]
    rr = rr_features(r.beat_samples, dcfg["local_rr_beats"])[keep]
    assert np.isfinite(rr).all()
    return {
        "x": normalize_beats(extract_windows(xf, centers, pre, post)).astype(np.float32),
        "rr": rr.astype(np.float32),
        "y": np.array([AAMI_CLASSES.index(c) for c in m["labels"][keep]], np.int8),
        "record": np.full(keep.sum(), rec, np.int16),
        "sample": centers.astype(np.int32),
        "n_drop_edge": int((m["in_aami"] & m["edge"]).sum()),
        "n_drop_no_rr": int((m["in_aami"] & m["no_rr"] & ~m["edge"]).sum()),
    }


def build_split(records: list[int], mitdb_dir: Path, dcfg: dict, out: Path) -> dict:
    parts = [process_record(r, mitdb_dir, dcfg) for r in records]
    arrays = {k: np.concatenate([p[k] for p in parts])
              for k in ("x", "rr", "y", "record", "sample")}
    np.savez(out, **arrays)
    return {"n_drop_edge": sum(p["n_drop_edge"] for p in parts),
            "n_drop_no_rr": sum(p["n_drop_no_rr"] for p in parts),
            "n_beats": int(len(arrays["y"]))}


def assign_folds(y: np.ndarray, record: np.ndarray, n_folds: int) -> dict[int, int]:
    """StratifiedGroupKFold over DS1 records, record -> fold.

    Stratifies on all kept beats, including the 8 Q beats. That is the
    assignment inspected and approved in Phase 2 (every fold >= 104 S beats);
    stratifying with Q removed happens to give a fold with only 38 S. The
    algorithm is greedy and deterministic (shuffle only breaks ties).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    folds = {}
    for k, (_, va) in enumerate(sgkf.split(y, y, record)):
        for r in np.unique(record[va]):
            folds[int(r)] = k
    assert sorted(folds) == sorted(DS1)
    return folds


def build_dataset(cfg: dict, root: Path) -> dict:
    data_dir = root / cfg["paths"]["data_dir"]
    cache = root / cfg["paths"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    mitdb = download(data_dir)
    check_split(mitdb)
    info = {"ds1": build_split(DS1, mitdb, cfg["data"], cache / "ds1.npz"),
            "ds2": build_split(DS2, mitdb, cfg["data"], cache / "ds2.npz")}
    d1 = np.load(cache / "ds1.npz")
    folds = assign_folds(d1["y"], d1["record"], cfg["cv"]["n_folds"])
    (cache / "folds.json").write_text(json.dumps(folds, indent=1))
    info["folds"] = folds
    return info


# ---------------------------------------------------------------- loading for training

def load_split(cfg: dict, root: Path, split: str, drop_q: bool = True) -> dict[str, np.ndarray]:
    d = dict(np.load(root / cfg["paths"]["cache_dir"] / f"{split}.npz"))
    if drop_q:
        m = d["y"] != Q_CODE
        d = {k: v[m] for k, v in d.items()}
    d["y"] = d["y"].astype(np.int64)
    return d


def load_folds(cfg: dict, root: Path) -> dict[int, int]:
    raw = json.loads((root / cfg["paths"]["cache_dir"] / "folds.json").read_text())
    return {int(k): v for k, v in raw.items()}


class RRScaler:
    """log, then z-score with mean/std fit on training beats only."""

    def fit(self, rr: np.ndarray) -> "RRScaler":
        z = np.log(rr)
        self.mean_, self.std_ = z.mean(0), z.std(0) + 1e-6
        return self

    def transform(self, rr: np.ndarray) -> np.ndarray:
        return ((np.log(rr) - self.mean_) / self.std_).astype(np.float32)
