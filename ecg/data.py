"""MIT-BIH loading, AAMI mapping, filtering, beat windows and RR features.

Everything in this module is per-record: nothing is fit across beats or across
patients. Statistics that are fit across beats (e.g. standardizing RR
features) live in the training code and are fit on training records only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

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

CLASSES = ["N", "S", "V", "F", "Q"]
AAMI = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
    **dict.fromkeys(["A", "a", "J", "S"], "S"),
    **dict.fromkeys(["V", "E"], "V"),
    "F": "F",
    **dict.fromkeys(["/", "f", "Q"], "Q"),
}
# Symbols that mark a QRS complex. Used for RR intervals, so a beat we don't
# classify (e.g. Q) still counts as the neighbour of the beats around it.
# '!' (ventricular flutter wave) and non-beat annotations are not QRS marks.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Window around the R-peak: 0.25 s before, 0.45 s after (P wave to T wave).
PRE = int(0.25 * FS)   # 90 samples
POST = int(0.45 * FS)  # 162 samples
WIN = PRE + POST       # 252 samples; R-peak at index PRE

LOCAL_RR_BEATS = 10  # local average RR over the preceding 10 intervals


def check_split() -> None:
    s1, s2, sp = set(DS1), set(DS2), set(PACED)
    assert len(s1) == len(DS1) == 22 and len(s2) == len(DS2) == 22
    assert not (s1 & s2) and not (s1 & sp) and not (s2 & sp)
    all48 = set(int(r) for r in wfdb.get_record_list("mitdb"))
    assert s1 | s2 | sp == all48, "DS1+DS2+paced must cover all 48 records"


MIRROR = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/"
CHECKSUMS = "https://physionet.org/files/mitdb/1.0.0/SHA256SUMS.txt"


def download(data_dir: Path) -> Path:
    """Fetch .hea/.dat/.atr for all 48 records and verify SHA-256.

    Files come from PhysioNet's open-data mirror on AWS (physionet.org itself
    is very slow to serve 150 files). Checksums come from physionet.org, so a
    corrupted or altered mirror file fails loudly.
    """
    import hashlib
    import urllib.request

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


@dataclass
class Record:
    name: int
    signal: np.ndarray       # MLII, mV, unfiltered
    beat_samples: np.ndarray  # R-peak sample index of every QRS annotation
    beat_symbols: np.ndarray  # original MIT-BIH symbol per QRS annotation
    n_samples: int


def load_record(rec: int, mitdb_dir: Path) -> Record:
    path = str(Path(mitdb_dir) / str(rec))
    sig, fields = wfdb.rdsamp(path)
    names = fields["sig_name"]
    if "MLII" not in names:
        raise ValueError(f"record {rec} has no MLII lead: {names}")
    x = sig[:, names.index("MLII")].astype(np.float64)
    ann = wfdb.rdann(path, "atr")
    sym = np.array(ann.symbol)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    return Record(rec, x, ann.sample[keep], sym[keep], len(x))


def load_annotations(rec: int, mitdb_dir: Path) -> Record:
    """Annotations and record length only (no signal), for fast counting."""
    path = str(Path(mitdb_dir) / str(rec))
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "atr")
    sym = np.array(ann.symbol)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    return Record(rec, np.empty(0), ann.sample[keep], sym[keep], hdr.sig_len)


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


def rr_features(beat_samples: np.ndarray, fs: int = FS) -> np.ndarray:
    """Per-beat RR features in seconds: pre, post, local average.

    pre/post use neighbouring QRS annotations of any type. Local average is
    the mean of up to the preceding LOCAL_RR_BEATS intervals (including pre).
    First beat has no pre-RR and last beat no post-RR: those are NaN and the
    beats are dropped downstream.
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
        j0 = max(0, i - LOCAL_RR_BEATS)
        local[i] = (csum[i] - csum[j0]) / (i - j0)
    return np.stack([pre, post, local], axis=1)


def aami_labels(symbols: np.ndarray) -> np.ndarray:
    """AAMI class per symbol, or '' for QRS types outside the mapping."""
    return np.array([AAMI.get(s, "") for s in symbols])


def beat_keep_mask(rec: Record) -> dict[str, np.ndarray]:
    """Which annotated beats survive, and why the others are dropped."""
    s = rec.beat_samples
    labels = aami_labels(rec.beat_symbols)
    in_aami = labels != ""
    edge = (s - PRE < 0) | (s + POST > rec.n_samples)
    no_rr = np.zeros(len(s), bool)
    no_rr[0] = no_rr[-1] = True  # first beat: no pre-RR; last: no post-RR
    keep = in_aami & ~edge & ~no_rr
    return {"labels": labels, "in_aami": in_aami, "edge": edge,
            "no_rr": no_rr, "keep": keep}


def extract_windows(x_filt: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx = centers[:, None] + np.arange(-PRE, POST)[None, :]
    return x_filt[idx]


def normalize_beats(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-beat z-score. Removes gain/offset differences between patients and
    electrode placements; uses only that beat, so it cannot leak."""
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return (w - mu) / (sd + eps)
