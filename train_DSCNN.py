import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_mfcc(X, y):

    augmented_X = []
    augmented_y = []

    for sample, label in zip(X, y):

        # ----------------------------------------------------
        # Original sample
        # ----------------------------------------------------

        augmented_X.append(sample)
        augmented_y.append(label)

        # ----------------------------------------------------
        # 1. Add small random noise
        # ----------------------------------------------------

        noise = np.random.normal(
            0,
            0.03,
            sample.shape
        ).astype(np.float32)

        noisy_sample = sample + noise

        augmented_X.append(noisy_sample)
        augmented_y.append(label)

        # ----------------------------------------------------
        # 2. Random time shift
        # ----------------------------------------------------

        shift = np.random.randint(-5, 6)

        shifted_sample = np.roll(
            sample,
            shift,
            axis=1
        )

        augmented_X.append(shifted_sample)
        augmented_y.append(label)

    return (
        np.array(augmented_X, dtype=np.float32),
        np.array(augmented_y, dtype=np.int64)
    )


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "features/X.npy"
LABEL_FILE = "features/y.npy"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "kws_dscnn_v1.keras"
)

NORMALIZATION_FILE = os.path.join(
    MODEL_DIR,
    "normalization_v2.npz"
)

CLASS_NAMES = [
    "keyword",
    "unknown",
    "noise"
]

EPOCHS = 50
BATCH_SIZE = 16


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("              DS-CNN TRAINING")
print("=" * 60)

print("\nLoading dataset...")

X = np.load(FEATURE_FILE)
y = np.load(LABEL_FILE)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ============================================================
# FIX FEATURE DIMENSIONS
# ============================================================

# We want:
#
# X = (number_of_samples, 13, 63)
#
# NOT:
#
# (number_of_samples, 1, 13, 1, 63)
#
# or other extra dimensions.

while X.ndim > 3:

    # Remove dimensions of size 1
    X = np.squeeze(X)

print("Corrected X shape:", X.shape)


# Make sure the final feature array is:
# (samples, MFCC, time)

if X.ndim != 3:

    raise ValueError(
        f"Expected X to have 3 dimensions "
        f"(samples, MFCC, time), but got {X.shape}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y == i)

    print(
        f"{class_name:10s}: {count}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


print("\nDataset split:")

print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# NORMALIZATION
# ============================================================

# Calculate mean/std for each MFCC coefficient.

mean = np.mean(
    X_train,
    axis=(0, 2),
    keepdims=True
)

std = np.std(
    X_train,
    axis=(0, 2),
    keepdims=True
)

std = np.maximum(
    std,
    1e-6
)


X_train = (
    X_train - mean
) / std

X_val = (
    X_val - mean
) / std

X_test = (
    X_test - mean
) / std


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Augment training data only.
#
# Validation and test data remain unchanged.

X_train, y_train = augment_mfcc(
    X_train,
    y_train
)

print("\nAfter augmentation:")

print("Training:", X_train.shape)
print("Labels:", y_train.shape)


# ============================================================
# SAVE NORMALIZATION
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

np.savez(
    NORMALIZATION_FILE,
    mean=mean,
    std=std
)

print(
    "\nNormalization saved:"
)

print(
    NORMALIZATION_FILE
)


# ============================================================
# ADD ONLY ONE CHANNEL DIMENSION
# ============================================================

# Before:
#
# (samples, 13, 63)
#
# After:
#
# (samples, 13, 63, 1)

X_train = X_train[..., np.newaxis]

X_val = X_val[..., np.newaxis]

X_test = X_test[..., np.newaxis]


print(
    "\nFinal training shape:",
    X_train.shape
)

print(
    "Final validation shape:",
    X_val.shape
)

print(
    "Final testing shape:",
    X_test.shape
)


# ============================================================
# BUILD DS-CNN
# ============================================================

model = tf.keras.Sequential([

    # --------------------------------------------------------
    # INITIAL FEATURE EXTRACTION
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # DS-CNN BLOCK 1
    # --------------------------------------------------------

    tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(
        32,
        (1, 1),
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # DS-CNN BLOCK 2
    # --------------------------------------------------------

    tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(
        64,
        (1, 1),
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel:")

model.build(
    input_shape=(
        None,
        X_train.shape[1],
        X_train.shape[2],
        X_train.shape[3]
    )
)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("                 STARTING TRAINING")
print("=" * 60)

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# TEST
# ============================================================

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print("\n" + "=" * 60)
print("                  TEST RESULTS")
print("=" * 60)

print(
    f"\nTest accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)

print("\nModel saved:")

print(
    MODEL_FILE
)

print("\nTraining complete!")