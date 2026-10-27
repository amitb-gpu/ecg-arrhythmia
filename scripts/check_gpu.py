"""Verify TensorFlow sees and actually computes on the GPU.

Fails loudly (non-zero exit) if there is no GPU, so training never silently
falls back to CPU. Checks: device list, compute capability, an fp32 matmul
placed on the GPU (checked against NumPy), an fp16 matmul (mixed precision
path), and one Conv1D training step (exercises cuDNN, which the model uses).
"""
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf


def main() -> int:
    print(f"TensorFlow {tf.__version__}, Keras {tf.keras.__version__ if hasattr(tf.keras, '__version__') else ''}")
    build = tf.sysconfig.get_build_info()
    print(f"Built with CUDA {build.get('cuda_version')}, cuDNN {build.get('cudnn_version')}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus}")
    if not gpus:
        print("FAIL: TensorFlow sees no GPU.")
        return 1
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"Device: {details.get('device_name')}, compute capability {details.get('compute_capability')}")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((4096, 4096), dtype=np.float32)
    b = rng.standard_normal((4096, 4096), dtype=np.float32)

    with tf.device("/GPU:0"):
        ta, tb = tf.constant(a), tf.constant(b)
        c = tf.matmul(ta, tb)  # warm-up / kernel load
        _ = c.numpy()
        t0 = time.perf_counter()
        for _ in range(20):
            c = tf.matmul(ta, tb)
        _ = c.numpy()
        dt = time.perf_counter() - t0
    print(f"fp32 matmul ran on: {c.device}")
    if "GPU" not in c.device:
        print("FAIL: matmul did not run on the GPU.")
        return 1
    tflops = 20 * 2 * 4096**3 / dt / 1e12
    # TF32 may be on by default, so compare with a loose relative tolerance.
    ref = a[:256] @ b
    rel_err = np.abs(c.numpy()[:256] - ref).max() / np.abs(ref).max()
    print(f"fp32 matmul: {tflops:.1f} TFLOP/s, max rel err vs NumPy {rel_err:.2e}")
    if rel_err > 1e-2:
        print("FAIL: GPU matmul result disagrees with NumPy.")
        return 1

    with tf.device("/GPU:0"):
        h = tf.matmul(tf.cast(ta, tf.float16), tf.cast(tb, tf.float16))
        _ = h.numpy()
    print(f"fp16 matmul ran on: {h.device}")

    # One Conv1D training step under mixed precision (cuDNN path).
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(252, 1)),
        tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(5, dtype="float32"),
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    x = rng.standard_normal((1024, 252, 1)).astype(np.float32)
    y = rng.integers(0, 5, 1024)
    hist = model.fit(x, y, batch_size=256, epochs=2, verbose=0)
    print(f"Conv1D mixed-precision fit OK, losses {[round(v, 4) for v in hist.history['loss']]}")
    print(f"Conv kernel variable on: {model.layers[0].kernel.value.device}")

    print("PASS: TensorFlow is computing on the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
