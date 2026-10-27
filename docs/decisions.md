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
