"""
Predict whether a single image is Real or Fake.

Usage:
    python predict.py <image_path>
"""

import os
import sys

import tensorflow as tf

from utils import preprocess_single_image, predict_image, CLASS_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_real_detector.h5")


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: File not found – {image_path}")
        sys.exit(1)

    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Processing {image_path} …")
    img = preprocess_single_image(image_path)
    label, confidence = predict_image(model, img)

    print(f"\n  Prediction : {label}")
    print(f"  Confidence : {confidence:.2%}")


if __name__ == "__main__":
    main()
