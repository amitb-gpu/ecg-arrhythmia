#!/usr/bin/env bash
# Create the `ecg` conda env and make TensorFlow find its GPU libraries.
#
# Why the activation hook: the TF 2.21 wheel's built-in library search path
# (RPATH) covers cublas/cudnn/cufft/cusparse but not cusolver, so TF reports
# "Cannot dlopen some GPU libraries" and silently registers no GPU. Adding the
# pip-installed nvidia/*/lib folders to LD_LIBRARY_PATH, only while this env is
# active, fixes it without touching the system CUDA install.
set -euo pipefail

ENV_NAME="${ENV_NAME:-ecg}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda run -n "$ENV_NAME" pip install -r "$HERE/requirements.txt"

PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"

cat > "$PREFIX/etc/conda/activate.d/ecg_cuda_libs.sh" <<'EOF'
export _ECG_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
_ecg_sp="$(python -c 'import site; print(site.getsitepackages()[0])')"
_ecg_libs="$(ls -d "$_ecg_sp"/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${_ecg_libs}${LD_LIBRARY_PATH:-}"
unset _ecg_sp _ecg_libs
# TF 2.21 has no prebuilt sm_120 (Blackwell) kernels; the driver JIT-compiles
# them from PTX on first use. A large cache means that happens only once.
export _ECG_OLD_CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-}"
export CUDA_CACHE_MAXSIZE=4294967296
EOF

cat > "$PREFIX/etc/conda/deactivate.d/ecg_cuda_libs.sh" <<'EOF'
export LD_LIBRARY_PATH="${_ECG_OLD_LD_LIBRARY_PATH:-}"
export CUDA_CACHE_MAXSIZE="${_ECG_OLD_CUDA_CACHE_MAXSIZE:-}"
unset _ECG_OLD_LD_LIBRARY_PATH _ECG_OLD_CUDA_CACHE_MAXSIZE
if [ -z "$LD_LIBRARY_PATH" ]; then unset LD_LIBRARY_PATH; fi
if [ -z "$CUDA_CACHE_MAXSIZE" ]; then unset CUDA_CACHE_MAXSIZE; fi
EOF

echo "Env '$ENV_NAME' ready. Verify with: conda run -n $ENV_NAME python scripts/check_gpu.py"
