"""
Train the Fake vs Real Image Detector CNN.
Two-phase training: (1) frozen base → (2) fine-tune top layers.

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
    unfreeze_top_layers,
    plot_training_history,
    evaluate_model,
    plot_confusion_matrix,
)

# ──────────────────────────── Configuration ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..")

# Actual data directories
FAKE_DIR = os.path.join(DATASET_DIR, "real_and_fake_face", "training_fake")
REAL_DIR = os.path.join(DATASET_DIR, "real_and_fake_face", "training_real")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fake_real_detector.h5")

PHASE1_EPOCHS = 3          # train classifier head only
PHASE2_EPOCHS = 10          # fine-tune top layers
BATCH_SIZE = 32
AUGMENTATION_FACTOR = 1
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def main():
    # ── 1. Load ALL data ─────────────────────────────────────────
    print("Loading dataset …")

    if not os.path.exists(REAL_DIR):
        print(f"\n🚨 [ACTION REQUIRED]: '{REAL_DIR}' does not exist!")
        return
    if not os.path.exists(FAKE_DIR):
        print(f"\n🚨 [ACTION REQUIRED]: '{FAKE_DIR}' does not exist!")
        return

    X_all, y_all = load_dataset(FAKE_DIR, REAL_DIR)
    print(f"  Total -> {len(X_all)} images  |  Fake: {int((y_all == 0).sum())}  Real: {int((y_all == 1).sum())}")

    # ── 2. Stratified Train / Val / Test split (70 / 15 / 15) ──
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=RANDOM_SEED, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"  Split Sizes:")
    print(f"    Train : {len(X_train)}  |  Fake: {int((y_train == 0).sum())}  Real: {int((y_train == 1).sum())}")
    print(f"    Val   : {len(X_val)}    |  Fake: {int((y_val == 0).sum())}  Real: {int((y_val == 1).sum())}")
    print(f"    Test  : {len(X_test)}   |  Fake: {int((y_test == 0).sum())}  Real: {int((y_test == 1).sum())}")

    # ── 3. Augment training data ─────────────────────────────────
    print(f"Augmenting training data (factor={AUGMENTATION_FACTOR}) …")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, factor=AUGMENTATION_FACTOR)
    print(f"  After augmentation: {len(X_train_aug)} training samples")

    # Shuffle
    idx = np.random.permutation(len(X_train_aug))
    X_train_aug, y_train_aug = X_train_aug[idx], y_train_aug[idx]

    # ── 4. Build model ───────────────────────────────────────────
    print("Building CNN (MobileNetV2 @ 224×224) …")
    model = build_cnn()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head with frozen base (fast convergence)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 55)
    print("  PHASE 1: Training classifier head (base frozen)")
    print("═" * 55)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    history_p1 = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze top layers for fine-tuning (slower, higher accuracy)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 55)
    print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("═" * 55)

    unfreeze_top_layers(model, num_layers_to_unfreeze=30)

    # Recompile with a much lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    history_p2 = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=PHASE2_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # ── 5. Combined training curves ──────────────────────────────
    # Merge histories for a unified plot
    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = history_p1.history[key] + history_p2.history[key]

    class CombinedHistory:
        def __init__(self, h):
            self.history = h

    plot_training_history(
        CombinedHistory(combined_history),
        save_path=os.path.join(BASE_DIR, "training_curves.png")
    )

    # ── 6. Evaluate SEPARATELY on Training and Testing sets ──────
    print("\n" + "═" * 55)
    print("  EVALUATING ON TRAINING SET (original, non-augmented)")
    print("═" * 55)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"  Training Loss:     {train_loss:.4f}")
    print(f"  Training Accuracy: {train_acc:.4f}  ({train_acc:.2%})")

    # Detailed training set report
    cm_train, report_train, _, _ = evaluate_model(model, X_train, y_train)
    plot_confusion_matrix(cm_train, save_path=os.path.join(BASE_DIR, "confusion_matrix_train.png"))

    print("\n" + "═" * 55)
    print("  EVALUATING ON TESTING SET")
    print("═" * 55)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Testing Loss:      {test_loss:.4f}")
    print(f"  Testing Accuracy:  {test_acc:.4f}  ({test_acc:.2%})")

    # Detailed testing set report
    cm_test, report_test, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(cm_test, save_path=os.path.join(BASE_DIR, "confusion_matrix_test.png"))

    # ── 7. Final Summary ─────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 53 + "╗")
    print("║           FINAL RESULTS SUMMARY                     ║")
    print("╠" + "═" * 53 + "╣")
    print(f"║  Training Accuracy:   {train_acc:.4f}  ({train_acc:.2%})            ║")
    print(f"║  Testing Accuracy:    {test_acc:.4f}  ({test_acc:.2%})            ║")
    print(f"║  Training Loss:       {train_loss:.4f}                          ║")
    print(f"║  Testing Loss:        {test_loss:.4f}                          ║")
    print("╠" + "═" * 53 + "╣")
    gap = abs(train_acc - test_acc)
    if gap > 0.10:
        print(f"║  ⚠️  Overfitting detected (gap: {gap:.2%})              ║")
    elif gap < 0.02:
        print(f"║  ✅ Excellent generalization (gap: {gap:.2%})            ║")
    else:
        print(f"║  ℹ️  Good generalization (gap: {gap:.2%})                ║")
    print("╚" + "═" * 53 + "╝")

    # Save summary to file
    summary_path = os.path.join(BASE_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FINAL RESULTS SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training Accuracy:  {train_acc:.4f} ({train_acc:.2%})\n")
        f.write(f"Testing Accuracy:   {test_acc:.4f} ({test_acc:.2%})\n")
        f.write(f"Training Loss:      {train_loss:.4f}\n")
        f.write(f"Testing Loss:       {test_loss:.4f}\n")
        f.write(f"Accuracy Gap:       {gap:.4f} ({gap:.2%})\n")
    print(f"\n📄 Summary saved to {summary_path}")
    print(f"📊 Training curves saved to training_curves.png")
    print(f"📊 Training confusion matrix saved to confusion_matrix_train.png")
    print(f"📊 Testing confusion matrix saved to confusion_matrix_test.png")
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")

    # ── 8. Save model (final) ────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Final model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
