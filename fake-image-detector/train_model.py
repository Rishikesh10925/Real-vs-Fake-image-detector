"""
Train the Fake vs Real Image Detector CNN.

Usage:
    python train_model.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils import (
    load_dataset,
    augment_dataset,
    build_cnn,
    plot_training_history,
    evaluate_model,
    plot_confusion_matrix,
)

# ──────────────────────────── Configuration ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# We will use the new "Train" and "Test" directories directly
DATASET_DIR = os.path.join(BASE_DIR, "..")

TRAIN_FAKE_DIR = os.path.join(DATASET_DIR, "Train", "Fake")
TRAIN_REAL_DIR = os.path.join(DATASET_DIR, "Train", "Real")

TEST_FAKE_DIR = os.path.join(DATASET_DIR, "Test", "Fake")
TEST_REAL_DIR = os.path.join(DATASET_DIR, "Test", "Real")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fake_real_detector.h5")

EPOCHS = 40
BATCH_SIZE = 32
AUGMENTATION_FACTOR = 2  # more augmentation to prevent overfitting
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def main():
    # ── 1. Load data ──────────────────────────────────────────────
    print("Loading Training dataset …")
    
    if not os.path.exists(TRAIN_REAL_DIR):
        print(f"\n🚨 [ACTION REQUIRED]: The folder '{TRAIN_REAL_DIR}' does not exist yet!")
        print("Please ensure your 'Real' images are placed inside the 'Train' folder before continuing.")
        return

    X_train, y_train = load_dataset(TRAIN_FAKE_DIR, TRAIN_REAL_DIR)
    print(f"  Train -> Loaded {len(X_train)} images  |  Fake: {int((y_train == 0).sum())}  Real: {int((y_train == 1).sum())}")

    print("Loading Testing dataset …")
    X_test_full, y_test_full = load_dataset(TEST_FAKE_DIR, TEST_REAL_DIR)
    print(f"  Test -> Loaded {len(X_test_full)} images  |  Fake: {int((y_test_full == 0).sum())}  Real: {int((y_test_full == 1).sum())}")

    # ── 2. Train / Val / Test split ───────────────────────────────
    # Since we have Train and Test explicitly, we'll split the Test further 
    # to get a small Validation set (50% validation, 50% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_full, y_test_full, test_size=0.5, random_state=RANDOM_SEED, stratify=y_test_full
    )
    print(f"  Final Split Sizes:")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # ── 3. Augment training data ──────────────────────────────────
    print("Augmenting training data …")
    X_train, y_train = augment_dataset(X_train, y_train, factor=AUGMENTATION_FACTOR)
    print(f"  After augmentation: {len(X_train)} training samples")

    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    # ── 4. Build model ────────────────────────────────────────────
    print("Building CNN …")
    model = build_cnn()
    
    # We use a much smaller learning rate (1e-4 instead of 1e-3) for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── 5. Train ──────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\nTraining …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Training curves ────────────────────────────────────────
    plot_training_history(history, save_path=os.path.join(BASE_DIR, "training_curves.png"))

    # ── 7. Evaluate on test set ───────────────────────────────────
    print("\nEvaluating on test set …")
    cm, report, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(cm, save_path=os.path.join(BASE_DIR, "confusion_matrix.png"))

    # ── 8. Save model ─────────────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
