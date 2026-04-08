"""
Utility functions for Fake vs Real Image Detector.
Handles image loading, preprocessing, augmentation, evaluation, and Grad-CAM.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────── Constants ────────────────────────────
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Fake", "Real"]


# ──────────────────────────── Data Loading ─────────────────────────
def load_images_from_folder(folder: str, label: int) -> tuple[list, list]:
    """Load images from a folder and assign a label (0 or 1)."""
    images, labels = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for fname in os.listdir(folder):
        if os.path.splitext(fname)[1].lower() not in valid_exts:
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)
    return images, labels


def load_dataset(fake_dir: str, real_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and merge fake/real datasets, normalize to [0, 1]."""
    fake_imgs, fake_labels = load_images_from_folder(fake_dir, label=0)
    real_imgs, real_labels = load_images_from_folder(real_dir, label=1)

    X = np.array(fake_imgs + real_imgs, dtype=np.float32) / 255.0
    y = np.array(fake_labels + real_labels, dtype=np.float32)
    return X, y


# ──────────────────────────── Augmentation ─────────────────────────
def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a single image (values in [0, 1])."""
    img = tf.image.random_brightness(image, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_flip_left_right(img)

    # Random rotation (up to ±15°)
    angle = tf.random.uniform([], -15, 15) * (np.pi / 180.0)
    img = tf.keras.preprocessing.image.apply_affine_transform(
        img.numpy(), theta=angle.numpy(), fill_mode="nearest"
    ) if tf.random.uniform([]) > 0.5 else img

    # Random zoom (0.9–1.1)
    if tf.random.uniform([]) > 0.5:
        zoom = tf.random.uniform([], 0.9, 1.1).numpy()
        h, w = IMG_SIZE
        new_h, new_w = int(h * zoom), int(w * zoom)
        img_np = img if isinstance(img, np.ndarray) else img.numpy()
        img_np = cv2.resize(img_np, (new_w, new_h))
        # Centre-crop or pad back to original size
        if zoom > 1.0:
            start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
            img_np = img_np[start_h:start_h + h, start_w:start_w + w]
        else:
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            img_np = cv2.copyMakeBorder(
                img_np, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_REFLECT,
            )
        img = img_np

    # Small Gaussian noise
    if tf.random.uniform([]) > 0.5:
        img_np = img if isinstance(img, np.ndarray) else img.numpy()
        noise = np.random.normal(0, 0.02, img_np.shape).astype(np.float32)
        img = np.clip(img_np + noise, 0.0, 1.0)

    img = np.array(img, dtype=np.float32) if not isinstance(img, np.ndarray) else img
    return np.clip(img, 0.0, 1.0)


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Augment each image `factor` times and append to the dataset."""
    aug_X, aug_y = [], []
    for i in range(len(X)):
        for _ in range(factor):
            aug_X.append(augment_image(X[i]))
            aug_y.append(y[i])
    aug_X = np.array(aug_X, dtype=np.float32)
    aug_y = np.array(aug_y, dtype=np.float32)
    return np.concatenate([X, aug_X]), np.concatenate([y, aug_y])


# ──────────────────────────── CNN Model ────────────────────────────
def build_cnn(input_shape: tuple = (*IMG_SIZE, 3)) -> tf.keras.Model:
    """Build a CNN using MobileNetV2 with Fine-Tuning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # UNFREEZE layers to allow fine-tuning for much higher accuracy
    base_model.trainable = True
    
    # Optional: Keep the first 100 layers frozen, fine-tune the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


# ──────────────────────────── Training Plots ───────────────────────
def plot_training_history(history, save_path: str | None = None):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved plot: {save_path}")
    plt.close(fig)


# ──────────────────────────── Evaluation ───────────────────────────
def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    """Print confusion matrix, precision, recall, F1 and return predictions."""
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test.astype(int), y_pred)
    report = classification_report(
        y_test.astype(int), y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    print("\n── Confusion Matrix ──")
    print(cm)
    print("\n── Classification Report ──")
    print(classification_report(y_test.astype(int), y_pred, target_names=CLASS_NAMES))

    return cm, report, y_pred, y_pred_prob


def plot_confusion_matrix(cm: np.ndarray, save_path: str | None = None):
    """Visualise confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved plot: {save_path}")
    plt.close(fig)


# ──────────────────────────── Prediction ───────────────────────────
def preprocess_single_image(image_path: str) -> np.ndarray:
    """Load, resize, and normalise a single image for prediction."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Preprocess an in-memory image (e.g. from Streamlit upload)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0


def predict_image(model: tf.keras.Model, image: np.ndarray) -> tuple[str, float]:
    """Run prediction on a preprocessed image array (H, W, 3) in [0, 1]."""
    img_batch = np.expand_dims(image, axis=0)
    prob = model.predict(img_batch, verbose=0)[0][0]
    label = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return label, float(confidence)


# ──────────────────────────── Grad-CAM ─────────────────────────────
def make_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray,
                         last_conv_layer_name: str | None = None) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image."""
    # Handle the MobileNetV2 base model nested inside the Sequential model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        last_conv_layer = base_model.output
        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer, model.get_layer(index=-1).output],
        )
        # We need a custom forward pass because of the nested structure
        img_batch = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            # 1. Pass through base_model
            conv_outputs = base_model(img_tensor, training=False)
            # 2. Pass through the rest of the layers
            x = conv_outputs
            for layer in model.layers[1:]:
                x = layer(x, training=False)
            predictions = x
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, 0]
    else:
        if last_conv_layer_name is None:
            # Auto-detect last Conv2D layer
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.outputs],
        )

        img_batch = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((img_array.shape[0], img_array.shape[1]))
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on an image (both in 0-1 float range)."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = alpha * heatmap_color + (1 - alpha) * image
    return np.clip(overlay, 0, 1)


# ──────────────────────────── Image Info ───────────────────────────
def get_image_info(file_bytes: bytes) -> dict:
    """Return metadata about an uploaded image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "resolution": f"{w} x {h}",
        "file_size_kb": round(len(file_bytes) / 1024, 2),
        "channels": c,
        "avg_brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "image_bgr": img,
        "image_gray": gray,
    }
