import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import joblib
import json
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Настройка страницы
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
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
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia Detection from Chest X-Ray</h1>
    <p>AI-Powered Diagnostic Assistant | ResNet18 + Genetic Algorithm</p>
</div>
""", unsafe_allow_html=True)

# Загрузка моделей
@st.cache_resource
def load_models():
    try:
        model = joblib.load('backend/model/pneumonia_model.pkl')
        with open('backend/model/selected_features.json', 'r') as f:
            feature_indices = json.load(f)
        with open('backend/model/scaler_params.json', 'r') as f:
            scaler_params = json.load(f)
        
        # CNN для извлечения признаков
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        cnn.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return model, feature_indices, scaler_params, cnn, transform
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure model files exist in 'backend/model/' directory")
        return None, None, None, None, None

# Загрузка
model, feature_indices, scaler_params, cnn, transform = load_models()

if model is not None:
    mean = np.array(scaler_params['mean'])
    std = np.array(scaler_params['std'])
    
    # Интерфейс
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a frontal chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with AI..."):
                    # Извлечение признаков
                    img_tensor = transform(image).unsqueeze(0)
                    
                    with torch.no_grad():
                        features = cnn(img_tensor).squeeze().numpy()
                    
                    # Выбор и нормализация признаков
                    features_selected = features[feature_indices]
                    features_normalized = (features_selected - mean) / (std + 1e-8)
                    features_normalized = features_normalized.reshape(1, -1)
                    
                    # Предсказание
                    prediction = model.predict(features_normalized)[0]
                    probabilities = model.predict_proba(features_normalized)[0]
                    
                    # Отображение результатов
                    if prediction == 1:
                        st.markdown("""
                        <div class="result-card pneumonia-result">
                            <h2 style="color: white;">⚠️ PNEUMONIA DETECTED</h2>
                            <p style="color: white; font-size: 1.2rem;">Confidence: {:.1f}%</p>
                            <p style="color: white;">Please consult a healthcare professional immediately.</p>
                        </div>
                        """.format(probabilities[1]*100), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-card normal-result">
                            <h2 style="color: #2c3e50;">✅ NORMAL</h2>
                            <p style="color: #2c3e50; font-size: 1.2rem;">Confidence: {:.1f}%</p>
                            <p style="color: #2c3e50;">No signs of pneumonia detected.</p>
                        </div>
                        """.format(probabilities[0]*100), unsafe_allow_html=True)
                    
                    # Метрики
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Normal Probability", f"{probabilities[0]*100:.1f}%")
                    with col_b:
                        st.metric("Pneumonia Probability", f"{probabilities[1]*100:.1f}%")
                    
                    # Прогресс-бар
                    st.progress(probabilities[1])
                    
                    # Детали модели
                    with st.expander("🔬 Model Details"):
                        st.markdown(f"""
                        - **Model:** Logistic Regression with GA-selected features
                        - **Features used:** {len(feature_indices)}/512 ({100 - len(feature_indices)/512*100:.1f}% reduction)
                        - **Inference time:** ~0.14ms
                        - **Accuracy:** 81.73%
                        """)
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        ### 🧠 AI Model Architecture
        
        1. **Feature Extraction**: ResNet18 CNN
        2. **Feature Selection**: Genetic Algorithm (260/512 features)
        3. **Classification**: Logistic Regression
        
        ### 📊 Performance Metrics
        - **Accuracy**: 81.73%
        - **F1-Score**: 80.15%
        - **Feature Reduction**: 49.2%
        - **Inference**: 0.14ms/image
        
        ### ⚠️ Disclaimer
        This tool is for research and educational purposes only.
        Not intended for clinical diagnosis. Always consult a healthcare professional.
        """)
        
        st.markdown("---")
        st.markdown("### 🔬 How It Works")
        st.markdown("""
        1. Upload chest X-ray image
        2. CNN extracts 512 features
        3. GA selects most informative features
        4. Model predicts pneumonia probability
        5. Results displayed with confidence score
        """)
    
    # Футер
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Developed with ❤️ using PyTorch, Scikit-learn, and Genetic Algorithms</p>", 
        unsafe_allow_html=True
    )
else:
    st.error("Failed to load models. Please check the model files.")