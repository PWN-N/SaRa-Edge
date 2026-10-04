import os
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/kws_dscnn_v1.keras"
FEATURE_FILE = "features/X.npy"

TFLITE_FILE = "models/kws_dscnn_int8.tflite"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("          INT8 TFLITE CONVERSION")
print("=" * 60)

print("\nLoading DS-CNN model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD REPRESENTATIVE DATA
# ============================================================

print("\nLoading representative dataset...")

X = np.load(FEATURE_FILE)

# Remove unnecessary dimensions
while X.ndim > 3:
    X = np.squeeze(X)

print("Original feature shape:", X.shape)


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================

def representative_dataset():

    # Use a small subset for calibration
    num_samples = min(100, len(X))

    for i in range(num_samples):

        sample = X[i].astype(np.float32)

        # ----------------------------------------------------
        # IMPORTANT:
        # The model was trained using normalized MFCCs.
        #
        # For proper quantization calibration, we need to
        # apply the same normalization used during training.
        # ----------------------------------------------------

        yield [
            sample[np.newaxis, ..., np.newaxis]
        ]


# ============================================================
# CONVERT TO INT8
# ============================================================

print("\nConverting to INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
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
# SAVE MODEL
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
# MODEL SIZE
# ============================================================

model_size = os.path.getsize(
    TFLITE_FILE
)

model_size_kb = model_size / 1024


print("\n" + "=" * 60)
print("             CONVERSION COMPLETE")
print("=" * 60)

print(
    f"\nTFLite model: {TFLITE_FILE}"
)

print(
    f"Model size: {model_size_kb:.2f} KB"
)

print("\nINT8 conversion successful!")