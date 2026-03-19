import os
import io
from datetime import datetime

import streamlit as st
import requests
import pandas as pd
from PIL import Image

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8000")

st.set_page_config(page_title="Refund Classifier", page_icon="📦", layout="wide")
st.title("📦 Refund Item Classification System")
st.markdown("Upload images of returned items for automated classification")

with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{MODEL_SERVICE_URL}/health", timeout=3).json()
        if health.get("model_loaded"):
            st.success("✅ Model Service: Online")
            st.info(f"Model Version: {health.get('model_version')}")
        else:
            st.error("❌ Model not loaded")
    except Exception:
        st.error("❌ Model Service: Offline")

    st.divider()
    st.markdown("**Categories**")
    st.markdown("Casual Shoes · Handbags · Shirts · Tops · Watches")

tab1, tab2 = st.tabs(["📤 Upload & Classify", "ℹ️ About"])

with tab1:
    st.header("Upload Images for Classification")

    uploaded_files = st.file_uploader(
        "Choose images (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} image(s) selected**")

        cols = st.columns(min(len(uploaded_files), 5))
        for idx, file in enumerate(uploaded_files[:5]):
            with cols[idx]:
                st.image(Image.open(file), caption=file.name, use_container_width=True)
        if len(uploaded_files) > 5:
            st.info(f"+ {len(uploaded_files) - 5} more images")

        if st.button("🚀 Run Classification", type="primary"):
            with st.spinner("Classifying..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), "image/jpeg"))
                        for f in uploaded_files
                    ]
                    resp = requests.post(
                        f"{MODEL_SERVICE_URL}/predict",
                        files=files_payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    predictions = resp.json()["predictions"]

                    st.success(f"✅ Done — {len(predictions)} image(s) classified")

                    df = pd.DataFrame(predictions)[["image_name", "predicted_class", "confidence"]]
                    df["confidence"] = df["confidence"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    st.bar_chart(
                        pd.DataFrame(predictions)["predicted_class"].value_counts()
                    )

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

with tab2:
    st.header("About This System")
    st.markdown("""
    ### 🎯 Purpose
    Automatically classifies returned e-commerce items to streamline warehouse operations
    and reduce manual sorting effort.

    ### 🏗️ Architecture
    - **Model Service**: FastAPI server with EfficientNet-B0 (96.53% test accuracy)
    - **This UI**: Streamlit frontend for manual batch classification

    ### 📋 Categories
    Casual Shoes · Handbags · Shirts · Tops · Watches

    ### 📊 Model Performance
    | Metric | Value |
    |--------|-------|
    | Test Accuracy | 96.53% |
    | Validation Accuracy | 98.40% |
    | Architecture | EfficientNet-B0 |
    | Training Data | 2,500 images (5 classes) |
    """)