import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="wide")

st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("ResNet18 + Genetic Algorithm + Logistic Regression")

# ---------- Загрузка моделей ----------
@st.cache_resource
def load_all():
    model = joblib.load("backend/model/model.pkl")
    scaler = joblib.load("backend/model/scaler.pkl")
    selected_features = joblib.load("backend/model/selected_features.pkl")

    cnn = models.resnet18(pretrained=True)
    cnn.fc = nn.Identity()
    cnn.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, scaler, selected_features, cnn, transform


model, scaler, selected_features, cnn, transform = load_all()

# ---------- Интерфейс ----------
uploaded_file = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):

            # 1. Извлечение признаков CNN
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                features = cnn(img_tensor).cpu().numpy().squeeze()

            # 2. Те же признаки, что при обучении
            features = features[selected_features].reshape(1, -1)

            # 3. Тот же scaler
            features = scaler.transform(features)

            # 4. Вероятность пневмонии
            proba = model.predict_proba(features)[0][1]

            # 5. Медицинский порог
            threshold = 0.35
            prediction = "PNEUMONIA" if proba > threshold else "NORMAL"

            # ---------- Вывод ----------
            st.subheader("Result")

            if prediction == "PNEUMONIA":
                st.error(f"⚠️ Pneumonia suspected\n\nProbability: {proba*100:.2f}%")
            else:
                st.success(f"✅ Normal\n\nProbability of pneumonia: {proba*100:.2f}%")

            st.progress(float(proba))

            st.markdown("---")
            st.write("### Probabilities")
            st.write(f"Normal: {(1-proba)*100:.2f}%")
            st.write(f"Pneumonia: {proba*100:.2f}%")
