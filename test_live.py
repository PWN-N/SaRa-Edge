import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf


# ==============================
# SETTINGS
# ==============================

MODEL_FILE = "models/kws_baseline.keras"

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


# ==============================
# LOAD MODEL
# ==============================

print("=" * 60)
print("             LIVE KWS TEST")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully.")


# ==============================
# RECORD AUDIO
# ==============================

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


# ==============================
# CONVERT TO 1D
# ==============================

audio = audio.flatten()


# ==============================
# EXTRACT MFCC
# ==============================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SAMPLE_RATE,
    n_mfcc=N_MFCC
)


# ==============================
# NORMALIZE
# ==============================

# Same normalization approach used during training
mean = np.mean(mfcc)
std = np.std(mfcc)

mfcc = (mfcc - mean) / (std + 1e-8)


# ==============================
# PREPARE INPUT
# ==============================

# Add batch dimension
# Add channel dimension

X = mfcc[np.newaxis, ..., np.newaxis]


# ==============================
# PREDICT
# ==============================

prediction = model.predict(
    X,
    verbose=0
)

predicted_class = np.argmax(prediction[0])

confidence = prediction[0][predicted_class] * 100


# ==============================
# DISPLAY RESULT
# ==============================

print("\n" + "=" * 60)
print("                 RESULT")
print("=" * 60)

print(f"\nPrediction : {CLASS_NAMES[predicted_class]}")
print(f"Confidence : {confidence:.2f}%")

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:10s}: "
        f"{prediction[0][i] * 100:.2f}%"
    )

print("=" * 60)