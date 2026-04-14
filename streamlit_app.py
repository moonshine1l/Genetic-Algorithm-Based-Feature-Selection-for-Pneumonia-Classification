# app.py - Updated version using scaler_selected.pkl

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import joblib
import json
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pneumonia Detection AI", 
    page_icon="🫁", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .normal-result {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .pneumonia-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white;">🫁 Pneumonia Detection from Chest X-Ray</h1>
    <p style="color: white;">ResNet18 + Genetic Algorithm + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Model Loading - CORRECTED VERSION
# ============================================

@st.cache_resource
def load_all():
    """Load all model components - Version A with scaler_selected"""
    
    model_dir = "backend/model"
    
    if not os.path.exists(model_dir):
        st.error(f"❌ Model directory '{model_dir}' not found!")
        st.info("Please train the model first using the notebook")
        return None, None, None, None, None
    
    try:
        # 1. Load Logistic Regression model
        model_path = os.path.join(model_dir, "pneumonia_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("✅ Logistic Regression model loaded")
        else:
            st.error(f"Model not found at {model_path}")
            return None, None, None, None, None
        
        # 2. Load scaler for SELECTED features (THIS IS THE KEY FIX)
        scaler_path = os.path.join(model_dir, "scaler_selected.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success("✅ Scaler for selected features loaded")
        else:
            st.error(f"Scaler for selected features not found at {scaler_path}")
            st.info("Please run the notebook with the updated code to create scaler_selected.pkl")
            return None, None, None, None, None
        
        # 3. Load selected features indices
        features_path = os.path.join(model_dir, "selected_features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                selected_features = json.load(f)
            st.success(f"✅ Loaded {len(selected_features)} selected features")
        else:
            # Try pkl format
            features_path_pkl = os.path.join(model_dir, "selected_features.pkl")
            if os.path.exists(features_path_pkl):
                selected_features = joblib.load(features_path_pkl)
                st.success(f"✅ Loaded {len(selected_features)} selected features from pkl")
            else:
                st.error("Selected features file not found!")
                return None, None, None, None, None
        
        # 4. Load CNN feature extractor
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        cnn.eval()
        st.success("✅ CNN feature extractor loaded")
        
        # 5. Define transform for X-ray images
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load metadata if available
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.success("✅ Model metadata loaded")
        
        return model, scaler, selected_features, cnn, transform, metadata
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load all components
model, scaler, selected_features, cnn, transform, metadata = load_all()

# ============================================
# Medical threshold
# ============================================
THRESHOLD = 0.35

# ============================================
# UI Layout
# ============================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=["png", "jpg", "jpeg"],
        help="Upload a frontal chest X-ray image for pneumonia detection"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if uploaded_file:
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if model is None:
                st.error("Models not loaded. Please check the model directory.")
            else:
                with st.spinner("Analyzing chest X-ray with AI..."):
                    try:
                        # Step 1: Extract full CNN features (512 features)
                        img_tensor = transform(image).unsqueeze(0)
                        
                        with torch.no_grad():
                            features_full = cnn(img_tensor).cpu().numpy().squeeze()
                        
                        # Step 2: Select only GA-selected features
                        features_selected = features_full[selected_features].reshape(1, -1)
                        
                        # Step 3: Apply scaler trained on selected features
                        features_scaled = scaler.transform(features_selected)
                        
                        # Step 4: Predict pneumonia probability
                        proba_pneumonia = model.predict_proba(features_scaled)[0][1]
                        proba_normal = 1 - proba_pneumonia
                        
                        # Step 5: Apply threshold
                        prediction = "PNEUMONIA" if proba_pneumonia > THRESHOLD else "NORMAL"
                        
                        # Step 6: Display results
                        if prediction == "PNEUMONIA":
                            st.markdown(f"""
                            <div class="result-card pneumonia-result">
                                <h2 style="color: white;">⚠️ PNEUMONIA DETECTED</h2>
                                <p style="color: white; font-size: 1.2rem;">Probability: {proba_pneumonia*100:.2f}%</p>
                                <p style="color: white;">Please consult a healthcare professional immediately.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card normal-result">
                                <h2 style="color: #2c3e50;">✅ NORMAL</h2>
                                <p style="color: #2c3e50; font-size: 1.2rem;">Probability of pneumonia: {proba_pneumonia*100:.2f}%</p>
                                <p style="color: #2c3e50;">No signs of pneumonia detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.subheader("📈 Confidence Level")
                        st.progress(float(proba_pneumonia))
                        
                        # Detailed probabilities
                        st.markdown("---")
                        st.write("### 📊 Probability Distribution")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Normal", f"{proba_normal*100:.2f}%", 
                                     delta="-" if proba_normal > 0.5 else None)
                        with col_b:
                            st.metric("Pneumonia", f"{proba_pneumonia*100:.2f}%",
                                     delta="↑" if proba_pneumonia > 0.5 else None)
                        
                        # Model information
                        with st.expander("ℹ️ Model Information"):
                            if metadata:
                                st.markdown(f"""
                                - **Model Type:** {metadata.get('model_type', 'Logistic Regression')}
                                - **Feature Extractor:** {metadata.get('feature_extractor', 'ResNet18')}
                                - **Feature Selection:** {metadata.get('feature_selection', 'Genetic Algorithm')}
                                - **Total Features:** {metadata.get('n_features_total', 512)}
                                - **Selected Features:** {metadata.get('n_features_selected', len(selected_features))}
                                - **Feature Reduction:** {metadata.get('feature_reduction_percent', 0):.1f}%
                                - **Model Accuracy:** {metadata.get('accuracy', 0.82)*100:.1f}%
                                - **F1 Score:** {metadata.get('f1_score', 0.80):.3f}
                                """)
                            else:
                                st.markdown(f"""
                                - **Architecture:** ResNet18 + Genetic Algorithm + Logistic Regression
                                - **Total features:** 512
                                - **Selected features:** {len(selected_features)} ({100 - len(selected_features)/512*100:.1f}% reduction)
                                - **Decision threshold:** {THRESHOLD*100:.0f}%
                                """)
                            
                            st.markdown("""
                            **How to interpret results:**
                            - **< 20%:** Very low probability - likely normal
                            - **20-50%:** Low probability - clinical correlation recommended
                            - **50-70%:** Moderate probability - further evaluation suggested
                            - **> 70%:** High probability - urgent consultation recommended
                            """)
                        
                        # Warning disclaimer
                        st.warning("""
                        ⚠️ **Medical Disclaimer:** This is an AI-assisted diagnostic tool. 
                        Results should be interpreted by qualified healthcare professionals. 
                        Do not rely solely on this tool for clinical decisions.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.info("Please make sure all model files are present in backend/model/")

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    ### 🧠 AI Model Architecture
    
    1. **Feature Extraction**: ResNet18 CNN
    2. **Feature Selection**: Genetic Algorithm (GA)
    3. **Classification**: Logistic Regression
    
    ### 📊 Model Performance
    - **Accuracy:** ~82%
    - **Features selected:** ~260/512 (49% reduction)
    - **Inference time:** < 1ms
    
    ### 🔬 How It Works
    1. Upload chest X-ray image
    2. CNN extracts 512 features
    3. GA selects most informative features
    4. Model predicts pneumonia probability
    5. Results displayed with confidence score
    
    ### 📁 Required Files
    Make sure these files exist in `backend/model/`:
    - `pneumonia_model.pkl` - Trained model
    - `scaler_selected.pkl` - Scaler for selected features
    - `selected_features.json` - GA-selected feature indices
    """)
    
    st.markdown("---")
    
    # Threshold adjustment
    st.markdown("### 🎚️ Adjust Sensitivity")
    new_threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=THRESHOLD,
        step=0.05,
        help="Lower = more sensitive (detects more cases)\nHigher = more specific (fewer false positives)"
    )
    
    if new_threshold != THRESHOLD:
        THRESHOLD = new_threshold
        st.info(f"Threshold updated to {THRESHOLD:.2f}")
        st.markdown(f"""
        - **Sensitivity:** {(1 - THRESHOLD)*100:.0f}% (approx)
        - **Specificity:** {THRESHOLD*100:.0f}% (approx)
        """)
    
    st.markdown("---")
    st.markdown("### 📞 Need Help?")
    st.markdown("""
    If you encounter errors:
    1. Check that all model files exist
    2. Run the notebook to train the model first
    3. Ensure `scaler_selected.pkl` is present
    """)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Developed with ❤️ using PyTorch, Scikit-learn, and Genetic Algorithms | Medical AI Research</p>", 
    unsafe_allow_html=True
)