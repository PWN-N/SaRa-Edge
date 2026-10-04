import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/kws_dscnn_v1.keras"

FEATURE_FILE = "features/X.npy"
LABEL_FILE = "features/y.npy"

NORMALIZATION_FILE = "models/normalization_v2.npz"

TFLITE_FILE = "models/kws_dscnn_int8_v2.tflite"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("       CORRECT INT8 TFLITE CONVERSION")
print("=" * 60)

print("\nLoading DS-CNN...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading MFCC features...")

X = np.load(FEATURE_FILE)
y = np.load(LABEL_FILE)

while X.ndim > 3:
    X = np.squeeze(X)

print("X shape:", X.shape)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\nLoading training normalization...")

normalization = np.load(
    NORMALIZATION_FILE
)

mean = normalization["mean"]
std = normalization["std"]

# Saved shape:
# (1, 13, 1)
#
# Convert to:
# (13, 1)

mean = mean[0]
std = std[0]

print("Mean shape:", mean.shape)
print("Std shape :", std.shape)


# ============================================================
# REPRESENTATIVE DATA
# ============================================================

# We use the SAME type of normalized MFCC data
# that the neural network sees during training.

X_normalized = (
    X - mean
) / std


# Add channel dimension

X_normalized = X_normalized[
    ..., np.newaxis
]

print(
    "Normalized representative shape:",
    X_normalized.shape
)


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================

def representative_dataset():

    # Use up to 100 representative samples

    num_samples = min(
        100,
        len(X_normalized)
    )

    for i in range(num_samples):

        sample = X_normalized[i]

        sample = sample[
            np.newaxis,
            ...
        ]

        yield [sample.astype(np.float32)]


# ============================================================
# CONVERT
# ============================================================

print("\nConverting to INT8...")

converter = (
    tf.lite.TFLiteConverter.from_keras_model(
        model
    )
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(TFLITE_FILE),
    exist_ok=True
)

with open(
    TFLITE_FILE,
    "wb"
) as f:

    f.write(tflite_model)


# ============================================================
# SIZE
# ============================================================

size = os.path.getsize(
    TFLITE_FILE
)

print("\n" + "=" * 60)
print("             CONVERSION COMPLETE")
print("=" * 60)

print(
    f"\nTFLite file:"
    f" {TFLITE_FILE}"
)

print(
    f"\nModel size:"
    f" {size / 1024:.2f} KB"
)

print("\nINT8 conversion successful.")