# Decision log

Every non-obvious choice, why it was made, and what the alternative was.
Grows phase by phase; the README summarizes it.

## Phase 1: environment

| Decision | Why | Alternative considered |
|---|---|---|
| Conda env `ecg`, Python 3.12 | Isolated from base; 3.12 is supported by TF 2.21 | 3.11 works too; no reason to prefer it |
| TensorFlow 2.21.0 via `tensorflow[and-cuda]` | Newest TF; pulls pip CUDA 12.9 / cuDNN 9.27, which include native Blackwell kernels for cuBLAS/cuDNN. No system CUDA toolkit needed | Conda-forge TF build, NVIDIA NGC container; both heavier for no gain here |
| Activation hook adds `nvidia/*/lib` to `LD_LIBRARY_PATH` | TF 2.21's wheel RPATH omits cusolver, so TF failed to load `libcusolver.so.11` and **silently registered no GPU**. Hook is scoped to this env only | Symlinking libs into TF's folder (fragile, breaks on reinstall) |
| `CUDA_CACHE_MAXSIZE=4 GB` | TF 2.21 has no prebuilt sm_120 kernels; the driver JIT-compiles them from PTX and caches the result. A big cache means it compiles once | Building TF from source with sm_120 (hours, no accuracy benefit) |
| `scripts/check_gpu.py` exits non-zero without a GPU | The failure mode above is silent: training would just run on CPU. The check asserts placement and numerics, and runs a mixed-precision Conv1D step to exercise cuDNN | Trusting `list_physical_devices` alone (doesn't prove kernels run) |
| `requirements-lock.txt` from `pip freeze` | Exact versions for reproducibility; `requirements.txt` stays human-readable | — |

Verified 2026-10-27: RTX 5090 seen as compute capability 12.0; fp32 matmul on
GPU matches NumPy (max rel err 2.8e-4, consistent with TF32); fp16 matmul and
mixed-precision Conv1D training step run on GPU.

## Phase 2: data pipeline (in progress)

### Data source and split
| Decision | Why | Alternative considered |
|---|---|---|
| Download from PhysioNet's AWS open-data mirror, verify SHA-256 against `physionet.org/files/mitdb/1.0.0/SHA256SUMS.txt` | physionet.org served ~1 file/min; mirror took 80 s for all 144 files. Checksums from the primary site make the mirror trustworthy | `wfdb.dl_database` (correct but ~2 h) |
| DS1/DS2 lists as given | Verified against Guo, Sim & Matuszewski 2019 (arXiv:1810.04121, both lists exact) and Mousavi et al. 2019 (arXiv:1812.07421, DS2 exact; their DS1 has a "101, 101" typo). Huang et al. 2014 also has typos in the text. `check_split()` asserts DS1, DS2 and paced are disjoint and cover all 48 records | — |
| DS1 class totals cross-check | Ours match Huang et al. 2014 Table 1 to within 2 beats per class (N 45,866 vs 45,868; S 944 vs 942; V 3,788 vs 3,787; F 415; Q 8) | — |

### Preprocessing (all per-record, so no cross-patient leakage)
| Decision | Why |
|---|---|
| Lead MLII; present in all DS1 records | Standard lead in the literature; closest to a single-lead wearable view |
| Butterworth bandpass 0.5-40 Hz, order 2, zero-phase (`sosfiltfilt`) | 0.5 Hz removes baseline wander; 40 Hz removes 60 Hz mains and EMG while keeping QRS energy (5-25 Hz). Zero-phase so R-peaks don't shift. Offline only: a streaming device would need a causal filter |
| Window -250 ms / +450 ms (90 + 162 = 252 samples), R-peak at index 90 | Covers P wave (S beats differ mainly in P-wave timing/shape) through T wave. Sanity plot `results/ds1_example_beats.png` confirms centering |
| Per-beat z-score | Removes gain/offset differences between patients and electrodes using only that beat. Trade-off: discards absolute amplitude, which the RR features partly compensate for |
| RR features: pre-RR, post-RR, local average (mean of preceding 10 RR) | S beats are premature: their defining signal is timing, not shape. RRs computed from all QRS annotations, including Q beats |
| Drop beats whose window crosses the record edge (19 in DS1) and first/last beats with no pre/post-RR (25 more) | 44 of 51,021 DS1 beats (0.09%): 42 N, 1 S, 1 F |
| Cross-beat statistics (RR standardization) fit on training records only, inside each fold | Fitting on all of DS1 would leak validation patients' heart-rate distribution into training |

### Q class: excluded from training and metrics (proposed)
DS1 has 8 Q beats from 3 records (101: 2, 203: 4, 208: 2). That is too few to
learn or to estimate a sensitivity with any precision (with 8 beats, a Wilson
95% CI is up to ±29 pp wide, e.g. 4/8 -> 22-78%). Decided from DS1 counts and common practice; DS2 Q counts will be
reported but were not consulted. Q beats stay in the RR computation as
neighbours of other beats.

### Validation: grouped 5-fold CV over DS1 (proposed, pending approval)
Minority classes are concentrated in a few patients: record 208 holds 90% of
DS1's F beats, record 209 holds 41% of S beats. Consequences:

- **Any** patient-level validation set either contains 208 (training then
  has ~40 F beats) or not (validation then has <=42 F beats). F validation
  is essentially "how do we do on patient 208", whatever the scheme.
- A single 4-5 record split is one draw from a high-variance distribution;
  early stopping and temperature would be fit on ~4 patients.
- `GroupKFold` balances fold size, not classes: one fold had V = 2.
  `StratifiedGroupKFold` gave every fold >=104 S and >=554 V. (Its `shuffle`
  only breaks ties, so seeds 0-2 gave identical folds; the assignment is
  effectively deterministic.)

**Choice:** `StratifiedGroupKFold(k=5)` over DS1 records. Each fold model
early-stops on its own held-out records. Out-of-fold predictions cover all 22
DS1 patients and are used for the validation report (mean ± spread across
folds) and to fit the temperature. On DS2, the final prediction averages the
5 fold models' probabilities.

**Costs accepted:** 5x training (minutes on the 5090); the deployed model is
an ensemble; temperature is fit on single-model out-of-fold logits but
applied to an ensemble, which is usually less overconfident, so DS2
calibration is reported for both the ensemble and the individual fold models.

**Rejected: retrain on all of DS1 after CV.** It would leave no held-out
patients for early stopping or for fitting the temperature.
