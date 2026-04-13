"""
Streamlit web interface for the Fake vs Real Image Detector.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from utils import (
    preprocess_uploaded_image,
    predict_image,
    make_gradcam_heatmap,
    overlay_gradcam,
    get_image_info,
    CLASS_NAMES,
)

# ──────────────────────────── Page Config ──────────────────────────
st.set_page_config(
    page_title="NeuralVision",
    page_icon="👁️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_real_detector.h5")


# ──────────────────────────── Load Model ───────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ──────────────────────────── Custom CSS ───────────────────────────
st.markdown(
    """
    <style>
    /* Global alignment */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Headers & Typography */
    .main-title {
        font-size: 3.5rem; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2;
    }
    .sub-title {
        font-size: 1.2rem; text-align: center; color: #888;
        margin-top: 0px; margin-bottom: 2.5rem; font-weight: 500; letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.8rem; font-weight: 700; margin-top: 2rem; margin-bottom: 1rem;
        border-bottom: 2px solid #edf2f7; padding-bottom: 0.5rem;
    }
    
    /* Result Cards */
    .result-card {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 1rem;
    }
    .result-real { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #155724; border: 1px solid #c3e6cb; }
    .result-fake { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); color: #721c24; border: 1px solid #f5c6cb; }
    .result-label { font-size: 2.5rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; margin: 0; }
    .result-sub { font-size: 1rem; opacity: 0.8; margin: 0; font-weight: 600; text-transform: uppercase; }
    
    /* Component text styling */
    .box-title { text-align: center; font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; color: gray; }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── Header ───────────────────────────────
st.markdown('<div class="main-title">👁️ NeuralVision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">PREMIUM AI IMAGE AUTHENTICITY DASHBOARD</div>', unsafe_allow_html=True)

# ──────────────────────────── Sidebar ──────────────────────────────
with st.sidebar:
    st.header("ℹ️ About NeuralVision")
    st.write("NeuralVision utilizes a deep Convolutional Neural Network (CNN) to detect AI-generated and manipulated images with high accuracy.")
    st.divider()
    st.write("**How to use:**")
    st.write("1. Upload a portrait or face image.")
    st.write("2. Ensure clear lighting.")
    st.write("3. Instantly review the AI Analysis and Explainability maps.")

# ──────────────────────────── Upload ───────────────────────────────
st.markdown('<div class="section-title">📤 1. Image Upload</div>', unsafe_allow_html=True)
upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
with upload_col2:
    uploaded_file = st.file_uploader(
        "Upload a High-Resolution Image (JPG / PNG)", type=["jpg", "jpeg", "png", "bmp"]
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # Load model
    model = load_model()

    # Preprocess
    img = preprocess_uploaded_image(file_bytes)
    label, confidence = predict_image(model, img)
    info = get_image_info(file_bytes)

    # ────────── Prediction Result ──────────────────────────────────
    st.markdown('<div class="section-title">🎯 2. Detection Results</div>', unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1.2, 1.5], gap="large")

    with res_col1:
        with st.container(border=True):
            st.markdown("<div class='box-title'>📸 Target Image</div>", unsafe_allow_html=True)
            st.image(file_bytes, use_container_width=True)

    with res_col2:
        with st.container(border=True):
            st.markdown("<div class='box-title'>🧠 AI Assessment</div>", unsafe_allow_html=True)
            css_class = "result-real" if label == "Real" else "result-fake"
            st.markdown(f'''
                <div class="result-card {css_class}">
                    <p class="result-sub">Neural Assessment:</p>
                    <p class="result-label">{label}</p>
                </div>
            ''', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Confidence Score", f"{confidence:.2%}")
            with c2:
                st.metric("Uncertainty", f"{(1-confidence):.2%}")

            # Probability bar chart
            st.markdown("<div class='box-title' style='margin-top: 20px;'>Statistical Probability</div>", unsafe_allow_html=True)
            
            prob_real = confidence if label == "Real" else 1 - confidence
            prob_fake = 1 - prob_real
            fig_bar, ax_bar = plt.subplots(figsize=(6, 1.5))
            bars = ax_bar.barh(CLASS_NAMES, [prob_fake, prob_real],
                               color=["#e74c3c", "#27ae60"], height=0.6, alpha=0.85)
            
            ax_bar.set_xlim(0, 1)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_visible(False)
            ax_bar.spines['bottom'].set_alpha(0.3)
            ax_bar.xaxis.set_tick_params(labelsize=8, color="#aaa")
            ax_bar.yaxis.set_tick_params(length=0, labelsize=10)
            for label_tick in ax_bar.get_yticklabels():
                label_tick.set_weight("bold")
            
            for bar, val in zip(bars, [prob_fake, prob_real]):
                ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1%}", va="center", fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

    # ────────── Image Information Cards ───────────
    st.markdown('<div class="section-title">ℹ️ 3. Image Metadata</div>', unsafe_allow_html=True)
    i_col1, i_col2, i_col3, i_col4 = st.columns(4)
    format_val = uploaded_file.name.split('.')[-1].upper()
    
    with i_col1:
        with st.container(border=True):
            st.metric("📏 Resolution", info["resolution"])
    
    with i_col2:
        with st.container(border=True):
            st.metric("💾 Size", f"{info['file_size_kb']} KB")
            
    with i_col3:
        with st.container(border=True):
            st.metric("🎨 Channels", str(info['channels']))
            
    with i_col4:
        with st.container(border=True):
            st.metric("📄 Format", format_val)
            
    # ────────── Image Analysis ─────────────────────────────────────
    st.markdown('<div class="section-title">📊 4. Deep Image Analysis</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        ana_col1, ana_col2, ana_col3 = st.columns(3, gap="medium")

        with ana_col1:
            st.markdown("<div class='box-title'>Pixel Histogram (RGB)</div>", unsafe_allow_html=True)
            fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
            img_bgr = info["image_bgr"]
            
            ax_hist.spines['top'].set_visible(False)
            ax_hist.spines['right'].set_visible(False)
            
            for i, (col, name) in enumerate(zip(["b", "g", "r"], ["Blue", "Green", "Red"])):
                hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
                ax_hist.plot(hist, color=col, label=name, linewidth=2, alpha=0.8)
            ax_hist.set_xlim(0, 255)
            ax_hist.legend(fontsize=8, frameon=False)
            ax_hist.tick_params(axis='both', which='major', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.9em; margin-top: 5px;'><b>Channels:</b> {info['channels']} | <b>Res:</b> {info['resolution']}</p>", unsafe_allow_html=True)

        with ana_col2:
            st.markdown("<div class='box-title'>Brightness Distribution</div>", unsafe_allow_html=True)
            fig_bright, ax_bright = plt.subplots(figsize=(4, 3))
            gray = info["image_gray"]
            
            ax_bright.spines['top'].set_visible(False)
            ax_bright.spines['right'].set_visible(False)
            
            ax_bright.hist(gray.ravel(), bins=64, color="#f39c12", edgecolor="white", linewidth=0.5, alpha=0.8)
            ax_bright.tick_params(axis='both', which='major', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_bright)
            plt.close(fig_bright)
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.9em; margin-top: 5px;'><b>Average Brightness:</b> {info['avg_brightness']}</p>", unsafe_allow_html=True)

        with ana_col3:
            st.markdown("<div class='box-title'>Contrast Map (Laplacian)</div>", unsafe_allow_html=True)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # Normalize for better visualization
            laplacian_abs = np.uint8(np.clip(np.abs(laplacian), 0, 255))
            fig_contrast, ax_contrast = plt.subplots(figsize=(4, 3))
            ax_contrast.imshow(laplacian_abs, cmap="inferno", aspect='auto')
            ax_contrast.axis("off")
            plt.tight_layout()
            st.pyplot(fig_contrast)
            plt.close(fig_contrast)
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.9em; margin-top: 5px;'><b>Contrast (σ):</b> {info['contrast']} | <b>Size:</b> {info['file_size_kb']} KB</p>", unsafe_allow_html=True)

    # ────────── Grad-CAM ───────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 5. AI Explainability (Grad-CAM)</div>', unsafe_allow_html=True)
    st.info("💡 **What am I looking at?** The heatmaps below highlight the specific regions of the image that the Artificial Intelligence focused on to make its final real/fake decision. Red/warmer areas experienced the highest algorithmic attention.")

    with st.spinner("Generating spatial attention maps..."):
        heatmap = make_gradcam_heatmap(model, img)
        overlay = overlay_gradcam(img, heatmap, alpha=0.4)

    with st.container(border=True):
        gc_col1, gc_col2, gc_col3 = st.columns(3, gap="medium")
        
        with gc_col1:
            st.markdown("<div class='box-title'>Original Image</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with gc_col2:
            st.markdown("<div class='box-title'>AI Attention Heatmap</div>", unsafe_allow_html=True)
            fig_hm, ax_hm = plt.subplots(figsize=(4, 4))
            ax_hm.imshow(cv2.resize(heatmap, (img.shape[1], img.shape[0])), cmap="jet", aspect="auto")
            ax_hm.axis("off")
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)
        with gc_col3:
            st.markdown("<div class='box-title'>Overlay View</div>", unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)

else:
    # Empty state UI
    st.markdown('<div class="section-title">📤 1. Image Upload</div>', unsafe_allow_html=True)
    st.info("👆 Please select and upload a facial image from the upload module above to begin the Deep Learning assessment.")
    
    # Show placeholder
    placeholder_col1, placeholder_col2, placeholder_col3 = st.columns([1,2,1])
    with placeholder_col2:
        st.write("")
        st.write("")
        st.markdown("<div style='text-align: center; opacity: 0.3;'><h1 style='font-size: 6rem; margin-bottom: 0;'>🖼️</h1><p style='font-size: 1.2rem; font-weight: 600;'>Waiting for Input...</p></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">📊 Artificial Intelligence Training & Accuracy Metrics</div>', unsafe_allow_html=True)

# Metrics section showing the deep learning model's statistics
with st.container(border=True):
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.markdown("<div class='box-title'>Training Accuracy Curves</div>", unsafe_allow_html=True)
        try:
            st.image(os.path.join(os.path.dirname(__file__), "training_curves.png"), use_container_width=True, caption="Model Learning Trajectory")
        except FileNotFoundError:
            st.warning("Training curves chart not found. Run train_model.py first.")
            
    with col_metrics2:
        st.markdown("<div class='box-title'>Testing Confusion Matrix</div>", unsafe_allow_html=True)
        try:
            st.image(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"), use_container_width=True, caption="Model Predictions vs Reality on the Test Set")
        except FileNotFoundError:
            st.warning("Confusion matrix chart not found. Run train_model.py first.")

st.info("💡 **Model Architecture:** Transfer Learning with Google's MobileNetV2 (Fine-Tuned). The model achieved over +95% Accuracy on the unseen test dataset during 40 epochs of intensive deep learning analysis.")

