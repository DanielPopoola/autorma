import streamlit as st
import requests
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import shutil
from datetime import datetime
import subprocess

# Config
MODEL_SERVICE_URL = "http://localhost:8000"
UPLOAD_DIR = Path("data/inference/input")
OUTPUT_DIR = Path("data/inference/output")

st.set_page_config(
    page_title="Refund Classifier",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Refund Item Classification System")
st.markdown("Upload images of returned items for automated classification")

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    try:
        health = requests.get(f"{MODEL_SERVICE_URL}/health", timeout=2).json()
        if health.get("model_loaded"):
            st.success("✅ Model Service: Online")
            st.info(f"Model Version: {health.get('model_version')}")
        else:
            st.error("❌ Model not loaded")
    except:
        st.error("❌ Model Service: Offline")
    
    st.divider()
    
    st.header("Statistics")
    checkpoint_file = Path("data/inference/checkpoints/checkpoint.json")
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        st.metric("Total Processed", len(checkpoint.get("processed_images", [])))
        if checkpoint.get("last_run"):
            st.caption(f"Last run: {checkpoint['last_run'][:19]}")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Classify", "📊 Results History", "ℹ️ About"])

with tab1:
    st.header("Upload Images for Classification")
    
    uploaded_files = st.file_uploader(
        "Choose images (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} images uploaded**")
        
        # Show preview grid
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:10]):  # Show first 10
            with cols[idx % 5]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_container_width=True)
        
        if len(uploaded_files) > 10:
            st.info(f"+ {len(uploaded_files) - 10} more images")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                with st.spinner("Processing images..."):
                    # Save uploaded files
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    saved_paths = []
                    
                    for file in uploaded_files:
                        file_path = UPLOAD_DIR / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        saved_paths.append(str(file_path.absolute()))
                    
                    # Run batch inference
                    try:
                        result = subprocess.run(
                            ["python", "orchestrator/batch_inference.py"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Classification complete!")
                            
                            # Load latest results
                            result_files = sorted(OUTPUT_DIR.glob("predictions_*.json"))
                            if result_files:
                                with open(result_files[-1]) as f:
                                    results = json.load(f)
                                
                                st.metric("Images Processed", results["successful"])
                                st.metric("Duration", f"{results['duration_seconds']:.2f}s")
                                
                                # Show results table
                                if results["predictions"]:
                                    df = pd.DataFrame(results["predictions"])
                                    df = df[["image_name", "predicted_class", "confidence"]]
                                    df["confidence"] = df["confidence"].apply(lambda x: f"{x:.2%}")
                                    
                                    st.dataframe(
                                        df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    # Download button
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Results (CSV)",
                                        csv,
                                        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )
                        else:
                            st.error(f"❌ Processing failed: {result.stderr}")
                    
                    except subprocess.TimeoutExpired:
                        st.error("❌ Processing timeout (>5 minutes)")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

with tab2:
    st.header("Classification Results History")
    
    result_files = sorted(OUTPUT_DIR.glob("predictions_*.json"), reverse=True)
    
    if not result_files:
        st.info("No results yet. Upload and classify images to see history.")
    else:
        selected_result = st.selectbox(
            "Select a batch run",
            result_files,
            format_func=lambda x: x.stem.replace("predictions_", "")
        )
        
        if selected_result:
            with open(selected_result) as f:
                data = json.load(f)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Images", data["total_images"])
            col2.metric("Successful", data["successful"])
            col3.metric("Failed", data["failed"])
            col4.metric("Duration", f"{data['duration_seconds']:.2f}s")
            
            # Class distribution chart
            if data["predictions"]:
                df = pd.DataFrame(data["predictions"])
                
                st.subheader("Class Distribution")
                class_counts = df["predicted_class"].value_counts()
                st.bar_chart(class_counts)
                
                st.subheader("Detailed Results")
                display_df = df[["image_name", "predicted_class", "confidence", "processed_at"]]
                display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            if data["failed_images"]:
                st.subheader("❌ Failed Images")
                failed_df = pd.DataFrame(data["failed_images"])
                st.dataframe(failed_df, use_container_width=True)

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ### 🎯 Purpose
    This system automatically classifies returned items into categories to streamline 
    warehouse operations and reduce manual sorting effort.
    
    ### 🏗️ Architecture
    - **Model Service**: FastAPI server with EfficientNet-B0 model (96.5% accuracy)
    - **Batch Orchestrator**: Automated nightly processing pipeline
    - **MLflow**: Model versioning and experiment tracking
    - **Prometheus + Grafana**: System monitoring and metrics
    
    ### 📋 Categories
    - Casual Shoes
    - Handbags
    - Shirts
    - Tops
    - Watches
    
    ### 🔄 Workflow
    1. Upload images via this UI or place in input folder
    2. System processes in batches of 10
    3. Results saved with confidence scores
    4. Automated cron job runs nightly at 2 AM
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Performance**")
        st.metric("Test Accuracy", "96.53%")
        st.metric("Training Time", "~12 minutes")
    
    with col2:
        st.markdown("**System Metrics**")
        st.metric("Avg Processing Time", "~5 seconds/batch")
        st.metric("Batch Size", "10 images")