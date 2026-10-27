#!/usr/bin/env bash
# End to end, from an empty data/ directory to every number and figure in the README.
#   bash scripts/run_all.sh
# Needs the `ecg` conda env (scripts/setup_env.sh). About 25 minutes on an
# RTX 5090, plus a one-time CUDA kernel compile on the first run.
set -euo pipefail
cd "$(dirname "$0")/.."
py() { conda run --no-capture-output -n ecg python "$@"; }

py scripts/check_gpu.py                                   # refuses to continue without a GPU
py scripts/build_dataset.py                               # download + verify MIT-BIH, build DS1/DS2 beats, folds
py scripts/inspect_ds1.py                                 # DS1-only counts and example-beat plot
py -m ecg.train --scheme inter                            # primary: patient-level 5-fold CV, early stopping
py -m ecg.train --scheme inter --fixed-epochs-from inter  # same, fixed epochs (early-stopping optimism)
py -m ecg.train --scheme intra_ds1                        # same, beat-level folds (leakage contrast)
py scripts/oof_summary.py                                 # DS1 out-of-fold tables

# DS2. The committed result came from one run on a clean tree at the frozen
# config; --overwrite reproduces it with the same frozen code.
if [[ -f results/run2/ds2/ds2_metrics.json ]]; then
  py -m ecg.evaluate --overwrite
else
  py -m ecg.evaluate
fi

py scripts/intra_patient_literature.py                    # random beat split over 44 records (trains on DS2 beats)
py scripts/make_figures.py                                # protocol comparison + DS2 error analysis
