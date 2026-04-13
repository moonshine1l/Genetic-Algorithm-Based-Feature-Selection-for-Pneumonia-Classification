from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import json
from PIL import Image
import io
import base64
from typing import Dict, Any
import logging

from utils.feature_extractor import FeatureExtractor

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(title="Pneumonia Detection API", 
              description="AI-powered pneumonia detection from chest X-rays",
              version="1.0.0")

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели и параметров
class ModelLoader:
    def __init__(self):
        self.model = None
        self.feature_indices = None
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_extractor = None
        self.load_models()
    
    def load_models(self):
        try:
            # Загрузка модели логистической регрессии
            self.model = joblib.load('model/pneumonia_model.pkl')
            logger.info("✅ Logistic Regression model loaded")
            
            # Загрузка индексов признаков
            with open('model/selected_features.json', 'r') as f:
                self.feature_indices = json.load(f)
            logger.info(f"✅ Loaded {len(self.feature_indices)} feature indices")
            
            # Загрузка параметров нормализации
            with open('model/scaler_params.json', 'r') as f:
                scaler_params = json.load(f)
                self.scaler_mean = np.array(scaler_params['mean'])
                self.scaler_std = np.array(scaler_params['std'])
            logger.info("✅ Scaler parameters loaded")
            
            # Инициализация экстрактора признаков
            self.feature_extractor = FeatureExtractor(device='cpu')
            logger.info("✅ Feature extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Предсказание для одного изображения"""
        # Извлечение признаков
        features_full = self.feature_extractor.extract_features(image)
        
        # Выбор признаков, отобранных GA
        features_selected = features_full[self.feature_indices]
        
        # Нормализация
        features_normalized = (features_selected - self.scaler_mean) / (self.scaler_std + 1e-8)
        features_normalized = features_normalized.reshape(1, -1)
        
        # Предсказание
        prediction = int(self.model.predict(features_normalized)[0])
        probabilities = self.model.predict_proba(features_normalized)[0]
        
        # Получение карты внимания
        attention_map = self.feature_extractor.get_attention_map(image)
        
        # Кодирование карты внимания в base64
        attention_map_normalized = (attention_map * 255).astype(np.uint8)
        attention_base64 = base64.b64encode(attention_map_normalized.tobytes()).decode('utf-8')
        
        return {
            'prediction': prediction,
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'normal': float(probabilities[0]),
                'pneumonia': float(probabilities[1])
            },
            'attention_map': attention_base64,
            'attention_shape': attention_map.shape
        }

# Инициализация загрузчика
model_loader = ModelLoader()

@app.get("/")
async def root():
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_loader.model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Предсказание пневмонии по загруженному изображению"""
    try:
        # Проверка формата файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Предсказание
        result = model_loader.predict(image_np)
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Пакетное предсказание для нескольких изображений"""
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = np.array(image)
            
            result = model_loader.predict(image_np)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})