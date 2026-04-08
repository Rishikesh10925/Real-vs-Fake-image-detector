# Fake vs Real Image Detector

A deep learning project that classifies face images as **Real** or **AI-Generated (Fake)** using a custom Convolutional Neural Network built with TensorFlow/Keras.

## Project Structure

```
fake-image-detector/
├── app.py                  # Streamlit web interface
├── train_model.py          # Model training script
├── predict.py              # CLI prediction script
├── utils.py                # Shared utilities (data loading, model, Grad-CAM, etc.)
├── requirements.txt        # Python dependencies
├── fake_real_detector.h5   # Saved model (after training)
├── training_curves.png     # Accuracy/loss plots (after training)
├── confusion_matrix.png    # Confusion matrix (after training)
└── README.md
```

## Dataset

The dataset is expected at `../real_and_fake_face/` relative to this folder:

```
real_and_fake_face/
├── training_fake/   (960 images)
└── training_real/   (1081 images)
```

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess images (resize to 128×128, normalise to 0–1)
- Apply data augmentation (brightness, contrast, rotation, zoom, flip, noise)
- Train a 4-block CNN for 25 epochs with Adam optimiser
- Display training accuracy/loss curves
- Evaluate on a held-out test set (confusion matrix, precision, recall, F1)
- Save the model as `fake_real_detector.h5`

### 2. Predict a Single Image (CLI)

```bash
python predict.py path/to/image.jpg
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Features of the web interface:
- **Image Upload** – drag-and-drop or browse
- **Prediction Result** – Real/Fake label, confidence score, probability bar chart
- **Image Information** – resolution, file size, channels, brightness, contrast
- **Image Analysis** – RGB histogram, brightness distribution, Laplacian contrast map
- **Model Explanation** – Grad-CAM heatmap highlighting influential regions

## Model Architecture

| Layer Block | Details |
|---|---|
| Block 1 | Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25) |
| Block 4 | Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25) |
| Head | Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid) |

## Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | Binary Cross-Entropy |
| Epochs | 25 (with early stopping, patience 5) |
| Batch Size | 32 |
| Image Size | 128 × 128 × 3 |

## Technologies

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Streamlit
