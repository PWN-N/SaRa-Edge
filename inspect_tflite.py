import tensorflow as tf
import os


MODEL_FILE = "models/kws_dscnn_int8_v2.tflite"


print("=" * 60)
print("          INT8 TFLITE MODEL INSPECTION")
print("=" * 60)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_FILE
)

interpreter.allocate_tensors()

print("Model loaded successfully.")


# ============================================================
# INPUT / OUTPUT DETAILS
# ============================================================

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\nINPUT DETAILS:")
print(input_details)


print("\nOUTPUT DETAILS:")
print(output_details)


# ============================================================
# MODEL SIZE
# ============================================================

size = os.path.getsize(
    MODEL_FILE
)

print("\nModel size:")
print(
    f"{size / 1024:.2f} KB"
)


# ============================================================
# IMPORTANT INFORMATION
# ============================================================

print("\nInput dtype:")
print(
    input_details[0]["dtype"]
)

print("\nInput shape:")
print(
    input_details[0]["shape"]
)

print("\nInput quantization:")
print(
    input_details[0]["quantization"]
)


print("\nOutput dtype:")
print(
    output_details[0]["dtype"]
)

print("\nOutput shape:")
print(
    output_details[0]["shape"]
)

print("\nOutput quantization:")
print(
    output_details[0]["quantization"]
)

print("              INSPECTION COMPLETE")