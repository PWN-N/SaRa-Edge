import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/kws_dscnn_v1.keras"

NORMALIZATION_FILE = "models/normalization_v2.npz"

SAMPLE_RATE = 16000
DURATION = 2
CHANNELS = 1
DEVICE = 1

N_MFCC = 13

CLASS_NAMES = [
    "keyword",
    "unknown",
    "noise"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("             LIVE KWS TEST")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD TRAINING NORMALIZATION
# ============================================================

print("\nLoading normalization parameters...")

normalization = np.load(
    NORMALIZATION_FILE
)

mean = normalization["mean"]
std = normalization["std"]

print("Normalization loaded successfully.")

print("Mean shape:", mean.shape)
print("Std shape :", std.shape)


# ============================================================
# RECORD AUDIO
# ============================================================

print("\nGet ready...")

sd.sleep(1500)

print(">>> SPEAK NOW <<<")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE
)

sd.wait()

print("Recording finished.")


# ============================================================
# CONVERT TO 1D
# ============================================================

audio = audio.flatten()


# ============================================================
# EXTRACT MFCC
# ============================================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SAMPLE_RATE,
    n_mfcc=N_MFCC
)

print("\nMFCC shape:", mfcc.shape)


# ============================================================
# CHECK MFCC SHAPE
# ============================================================

if mfcc.shape != (13, 63):

    raise ValueError(
        f"Unexpected MFCC shape: {mfcc.shape}. "
        f"Expected (13, 63)."
    )


# ============================================================
# NORMALIZE
# ============================================================

# Saved training normalization:
# mean shape = (1, 13, 1)
# std shape  = (1, 13, 1)

# Remove only the first dimension.
# Result:
# mean -> (13, 1)
# std  -> (13, 1)

mean_live = mean[0]
std_live = std[0]

mfcc = (
    mfcc - mean_live
) / std_live


# ============================================================
# PREPARE INPUT
# ============================================================

# MFCC:
# (13, 63)
#
# Add channel:
# (13, 63, 1)
#
# Add batch:
# (1, 13, 63, 1)

X = mfcc[..., np.newaxis]
X = X[np.newaxis, ...]

print("Model input shape:", X.shape)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    X,
    verbose=0
)

predicted_class = np.argmax(
    prediction[0]
)

confidence = (
    prediction[0][predicted_class] * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("                 RESULT")
print("=" * 60)

print(
    f"\nPrediction : "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence : "
    f"{confidence:.2f}%"
)

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:10s}: "
        f"{prediction[0][i] * 100:.2f}%"
    )

print("=" * 60)