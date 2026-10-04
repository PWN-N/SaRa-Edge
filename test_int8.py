import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/kws_dscnn_int8_v2.tflite"
FEATURE_FILE = "features/X.npy"
LABEL_FILE = "features/y.npy"

CLASS_NAMES = [
    "keyword",
    "unknown",
    "noise"
]


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(FEATURE_FILE)
y = np.load(LABEL_FILE)

while X.ndim > 3:
    X = np.squeeze(X)


# ============================================================
# CREATE SAME TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

normalization = np.load(
    "models/normalization_v2.npz"
)

mean = normalization["mean"]
std = normalization["std"]

mean = mean[0]
std = std[0]


# ============================================================
# NORMALIZE TEST DATA
# ============================================================

X_test = (
    X_test - mean
) / std


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_test = X_test[..., np.newaxis]


print("=" * 60)
print("             INT8 TFLITE TEST")
print("=" * 60)

print("\nTest shape:", X_test.shape)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

interpreter = tf.lite.Interpreter(
    model_path=MODEL_FILE
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)


output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


# ============================================================
# RUN INT8 INFERENCE
# ============================================================

y_pred = []


for sample in X_test:

    sample = sample[np.newaxis, ...]

    # --------------------------------------------
    # FLOAT → INT8
    # --------------------------------------------

    quantized_sample = (
        sample / input_scale
        + input_zero_point
    )

    quantized_sample = np.round(
        quantized_sample
    ).astype(np.int8)

    interpreter.set_tensor(
        input_index,
        quantized_sample
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    # --------------------------------------------
    # INT8 → class
    # --------------------------------------------

    predicted_class = np.argmax(
        output[0]
    )

    y_pred.append(
        predicted_class
    )


y_pred = np.array(y_pred)


# ============================================================
# RESULTS
# ============================================================

accuracy = np.mean(
    y_pred == y_test
)

print("\n" + "=" * 60)
print("                 RESULTS")
print("=" * 60)

print(
    f"\nINT8 Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


print("Confusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
