import os
import numpy as np
import librosa

DATASET_DIR = "dataset"
OUTPUT_DIR = "features"

CLASSES = [
    "keyword",
    "unknown",
    "noise"
]

SAMPLE_RATE = 16000
DURATION = 2
N_MFCC = 13

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []

print("=" * 60)
print("           MFCC FEATURE EXTRACTION")
print("=" * 60)

for label, class_name in enumerate(CLASSES):

    folder = os.path.join(DATASET_DIR, class_name)

    print(f"\nProcessing: {class_name}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    for filename in files:

        filepath = os.path.join(folder, filename)

        try:
            # Load audio
            audio, sr = librosa.load(
                filepath,
                sr=SAMPLE_RATE,
                mono=True
            )

            # Make every sample exactly 2 seconds
            target_length = SAMPLE_RATE * DURATION

            if len(audio) < target_length:
                audio = np.pad(
                    audio,
                    (0, target_length - len(audio))
                )
            else:
                audio = audio[:target_length]

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC
            )

            X.append(mfcc)
            y.append(label)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed: {len(files)} files")


# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Save features
np.save(
    os.path.join(OUTPUT_DIR, "X.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_DIR, "y.npy"),
    y
)

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nClass labels:")
print("0 = keyword")
print("1 = unknown")
print("2 = noise")

print("\nSaved:")
print("features/X.npy")
print("features/y.npy")