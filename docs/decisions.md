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

## Phase 2: data pipeline

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
| Plus two ratios: pre-RR / local average, post-RR / pre-RR | Prematurity relative to *this patient's* rhythm. A pre-RR of 0.6 s is premature at 60 bpm and normal at 100 bpm; the ratio carries that, which matters when test patients have heart rates the model never saw. Uses only the patient's own recent beats |
| All 5 RR features log-transformed, then z-scored with mean/std from the training fold's records | RRs are positive and right-skewed; log makes ratios symmetric around 0 (premature < 0 < compensatory pause) |
| Drop beats whose window crosses the record edge (19 in DS1) and first/last beats with no pre/post-RR (25 more) | 44 of 51,021 DS1 beats (0.09%): 42 N, 1 S, 1 F |
| Cross-beat statistics (RR standardization) fit on training records only, inside each fold | Fitting on all of DS1 would leak validation patients' heart-rate distribution into training |

### Q class: excluded from training and metrics (approved)
DS1 has 8 Q beats from 3 records (101: 2, 203: 4, 208: 2). That is too few to
learn or to estimate a sensitivity with any precision (with 8 beats, a Wilson
95% CI is up to ±29 pp wide, e.g. 4/8 -> 22-78%). Decided from DS1 counts and common practice; DS2 Q counts will be
reported but were not consulted. Q beats stay in the RR computation as
neighbours of other beats.

### Validation: grouped 5-fold CV over DS1 (approved)
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

**Calibration with the ensemble:** one temperature T is fit on the pooled
out-of-fold logits (all DS1 patients). On DS2, T is applied to each fold
model's logits, each is softmaxed, and the 5 probability vectors are averaged.
This keeps T operating on the same kind of input it was fit on (single-model
logits). DS2 calibration is reported for each single model and for the
ensemble, before and after T.

**Costs accepted:** 5x training (minutes on the 5090); the deployed model is
an ensemble of 5.

**Pre-declared primary model:** the early-stopped 5-fold ensemble. Any other
variant (e.g. fixed epochs) is compared on DS1 out-of-fold scores only, so DS2
is never used to choose between variants.

**Rejected: retrain on all of DS1 after CV.** It would leave no held-out
patients for early stopping or for fitting the temperature.

### Known biases and limitations (recorded before training)
- **Out-of-fold scores are slightly optimistic.** Each fold model's stopping
  epoch is chosen on the same held-out records that produce its out-of-fold
  score. For comparison, a second run trains every fold for a fixed epoch
  count (median best epoch across folds), which removes the per-fold
  selection. A single shared epoch count is still chosen from the same data,
  so even that run carries a small bias. DS2 is the unbiased estimate.
- **One-beat latency.** Post-RR means a beat is classified only after the next
  beat arrives (~0.6-1 s). Fine offline (Holter review); a real-time alarm
  would either wait one beat or drop post-RR and accept weaker S detection.
- **F results are effectively one patient.** Record 208 holds 90% of DS1's F
  beats. Per-fold F metrics are reported only for the fold that holds 208;
  for other folds F counts are 0-21, and their F scores are noise. DS2 F
  results are likewise driven by whichever DS2 records hold F beats.

### Built dataset (2026-10-27)
| Split | Records | N | S | V | F | Q |
|---|---|---|---|---|---|---|
| DS1 fold 0 | 114 118 208 215 | 8,760 | 113 | 1,215 | 377 | 2 |
| DS1 fold 1 | 101 116 119 209 | 8,318 | 387 | 554 | 0 | 2 |
| DS1 fold 2 | 106 108 115 201 205 230 | 11,649 | 135 | 807 | 15 | 0 |
| DS1 fold 3 | 112 203 207 220 | 8,556 | 204 | 654 | 1 | 4 |
| DS1 fold 4 | 109 122 124 223 | 8,541 | 104 | 558 | 21 | 0 |
| **DS1** | 22 | 45,824 | 943 | 3,788 | 414 | 8 |
| **DS2** (totals only) | 22 | 44,218 | 1,836 | 3,219 | 388 | 7 |

Dropped: DS1 19 edge + 25 no-RR; DS2 20 edge + 24 no-RR.

- Fold assignment stratifies on all kept beats including Q. Stratifying with
  Q removed (tried first) produced a fold with only 38 S beats; the approved
  assignment has >= 104 S in every fold. Chosen on DS1 class balance only.
- Record 208 is in fold 0, so F metrics are meaningful only for fold 0, and
  the fold-0 model trains on just 37 F beats.

## Phase 3: model and training

### Model (`ecg/model.py`, 186,628 parameters)
| Decision | Why | Alternative considered |
|---|---|---|
| Two inputs: beat waveform (252 x 1) and 5 RR features | Morphology separates V and F; timing separates S. A waveform-only CNN cannot see the neighbouring beats that define prematurity | Longer multi-beat window (lets the CNN see RR implicitly, but triples input and mixes in neighbour morphology) |
| Conv stem (32, k=7) + 3 residual blocks (32/64/128, k=5, stride 2 after the first) + global average pooling | Receptive field covers the QRS (~100 ms = 36 samples) and the T wave by the last block. Residual connections make the depth trainable without tuning; GAP keeps parameters low, which matters with ~40k training beats from 17-18 patients | Plain CNN, LSTM/attention (start simple; nothing yet shows it's needed) |
| RR branch Dense(16), concatenated, head Dense(64), dropout 0.3 | Lets RR interact non-linearly with morphology; dropout because the effective sample size is ~18 patients, not 40k beats | — |
| Output raw logits in float32 | Loss and softmax stay in full precision under mixed precision; temperature scaling needs logits | — |
| No data augmentation (first run) | Keep the baseline interpretable. Amplitude scaling / noise / baseline-wander augmentation is a natural next step for inter-patient robustness | — |

### Class imbalance: class-weighted cross-entropy
DS1 training folds are ~90% N, ~2% S, ~7% V, <1% F. Unweighted, the model can
reach ~90% accuracy predicting N for everything.

- **Choice:** "balanced" weights w_c = n / (K * n_c), computed from each
  training fold only. Every class contributes equally to the loss in
  expectation. In fold 0 (208 held out) only 37 F beats remain, so each F beat
  weighs ~1,000x an N beat (273.7 vs 0.27): that fold's F boundary is fit to a
  few dozen beats from a handful of patients.
- **Why not focal loss:** it adds a hyperparameter (gamma) that would have to
  be tuned on ~18 patients. It also deliberately produces under-confident
  probabilities, which muddies the calibration analysis. Weighted CE changes
  only the class prior, which is easy to reason about.
- **Why not oversampling:** with ~37-414 F beats, repeating them mostly
  teaches the model those specific beats (in effect, patient 208's beats).
  It is equivalent to weighting in expectation but noisier.
- **Calibration consequence (known in advance):** weighting trains the model
  as if classes were equally common, so its probabilities over-predict rare
  classes relative to true prevalence. A single temperature rescales
  confidence but cannot fix a prior shift. Phase 4 measures how much this
  matters; a principled fix (adding log-prior offsets to the logits) is noted
  as next work, since it trades off against sensitivity.

### Training
| Decision | Why |
|---|---|
| AdamW, lr 1e-3, weight decay 1e-4, batch 256, constant LR | Standard defaults; no LR schedule, so there is one less thing selected on validation data |
| Early stopping on **class-weighted validation loss**, patience 8, max 60 epochs; best checkpoint saved and reloaded | Macro-F1 is undefined on folds with no F beats (fold 1); weighted loss is defined everywhere and matches the training objective. Validation macro-F1 is logged for the curves only |
| Mixed precision (`mixed_float16`); Keras wraps the optimizer in `LossScaleOptimizer` automatically | fp16 math runs on tensor cores and halves activation memory; verified the policy and loss scaling are active in the training log. Speedup at this small model size was not measured |
| Seeds (`keras.utils.set_random_seed(seed + fold)`) + `enable_op_determinism()` | Verified: retraining fold 0 reproduces its validation logits bit for bit |
| Training refuses to start without a GPU | The silent-CPU failure from Phase 1 cannot recur |

### Leakage-contrast experiment on DS1 (`--scheme intra_ds1`)
Same model, data, folds count and budget, but beats are assigned to folds at
random (`StratifiedKFold` over beats), so every patient appears on both sides.
The gap between this run's out-of-fold scores and the patient-level run's is
the leakage effect, measured on identical data with DS2 untouched. The
literature-style intra-patient experiment (random beat split of all 44
records) runs in Phase 4, after the DS2 evaluation of the primary model is
frozen.

### Run 1 results (DS1 out-of-fold only; DS2 untouched) and diagnosis
| | Patient-level CV, early stop | Patient-level CV, fixed 4 epochs | Beat-level CV (leaky) |
|---|---|---|---|
| macro-F1 (N,S,V,F) | 0.423 | 0.459 | 0.797 |
| N Se / PPV | 0.659 / 0.992 | 0.832 / 0.986 | 0.962 / 0.999 |
| S Se / PPV | 0.544 / 0.078 | 0.347 / 0.093 | 0.966 / 0.528 |
| V Se / PPV | 0.848 / 0.566 | 0.938 / 0.582 | 0.975 / 0.877 |
| F Se / PPV | 0.872 / 0.044 | 0.256 / 0.039 | 0.957 / 0.437 |
| per-fold macro-F1 (N,S,V) | 0.570 ± 0.158 | 0.650 ± 0.150 | 0.866 ± 0.043 |
| best epochs | 8, 9, 1, 2, 4 | 4 (fixed) | 28, 22, 30, 24, 33 |

**Leakage gap:** 0.80 vs 0.42 macro-F1 on identical data and model. With
beat-level folds the model keeps improving for ~25-30 epochs, because
validation beats come from patients it is training on. With patient-level
folds, validation loss bottoms out within 1-9 epochs.

**Two flaws in my Run 1 design (found in diagnosis, not by looking at DS2):**
1. *Class weights explode when a fold loses its F patient.* Fold 0 holds
   out record 208, leaving 37 F training beats, so balanced weighting gives
   each F beat ~1,000x an N beat (other folds: ~90x). The fold-0 model
   predicts F for most held-out beats: N sensitivity 0.195; 7,447 N beats
   called F across all folds.
2. *Early stopping on class-weighted validation loss listens to a handful of
   patients.* Share of validation-loss weight from F: fold 0 92% (all
   from record 208), fold 4 9% from 21 beats. Fold 0's validation loss jumps
   between 4 and 25 epoch to epoch. In folds 2-4, validation loss rises
   while validation macro-F1 still improves: weighted loss penalizes growing
   confidence even when decisions get better. Fold 2 stopped at epoch 1.
   The fixed-epoch run beating early stopping (0.459 vs 0.423) shows the
   stopping rule was hurting, not just biasing the score upward.

Any fix is chosen after seeing Run 1's out-of-fold results, so Run 2's
out-of-fold scores carry that selection on top of the early-stopping bias.
DS2 remains the unbiased check.
