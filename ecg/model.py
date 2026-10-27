"""Compact two-input 1D residual CNN: beat waveform + RR features.

Waveform branch: conv stem, then one residual block per stage (stride 2 from
the second stage on), global average pooling. RR branch: a small dense layer.
The two are concatenated and passed to a small head producing logits.

Logits (not probabilities) come out in float32: under mixed precision the
softmax/loss stay in full precision, and temperature scaling needs logits.
"""
from __future__ import annotations

import keras
from keras import layers


def residual_block(x, filters: int, kernel: int, stride: int, wd: float, name: str):
    reg = keras.regularizers.l2(wd) if wd else None
    shortcut = x
    y = layers.Conv1D(filters, kernel, strides=stride, padding="same", use_bias=False,
                      kernel_regularizer=reg, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv1D(filters, kernel, padding="same", use_bias=False,
                      kernel_regularizer=reg, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, use_bias=False,
                                 kernel_regularizer=reg, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.ReLU(name=f"{name}_out")(y)


def build_model(mcfg: dict, win: int, n_rr: int, n_classes: int,
                weight_decay: float = 0.0) -> keras.Model:
    beat_in = keras.Input(shape=(win, 1), name="beat")
    rr_in = keras.Input(shape=(n_rr,), name="rr")

    x = layers.Conv1D(mcfg["stem_filters"], mcfg["stem_kernel"], padding="same",
                      use_bias=False, name="stem_conv")(beat_in)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    for i, f in enumerate(mcfg["block_filters"]):
        x = residual_block(x, f, mcfg["block_kernel"], stride=1 if i == 0 else 2,
                           wd=weight_decay, name=f"block{i}")
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    r = layers.Dense(mcfg["rr_units"], activation="relu", name="rr_dense")(rr_in)

    h = layers.Concatenate(name="fuse")([x, r])
    h = layers.Dropout(mcfg["dropout"], name="drop1")(h)
    h = layers.Dense(mcfg["head_units"], activation="relu", name="head_dense")(h)
    h = layers.Dropout(mcfg["dropout"], name="drop2")(h)
    logits = layers.Dense(n_classes, dtype="float32", name="logits")(h)
    return keras.Model([beat_in, rr_in], logits, name="ecg_resnet1d")
