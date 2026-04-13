import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import zoom

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Загрузка CNN для извлечения признаков
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Убираем классификатор
        self.cnn = self.cnn.to(device)
        self.cnn.eval()
        
        # Трансформация изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Регистрация хука для получения карт активации
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(output.detach())
        
        # Хук для layer4 (последний сверточный слой)
        layer = self.cnn._modules.get('layer4')
        if layer:
            layer.register_forward_hook(hook_fn)
    
    def extract_features(self, image):
        """Извлечение признаков из изображения"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.cnn(image_tensor)
        
        return features.cpu().numpy().flatten()
    
    def get_attention_map(self, image, target_size=(224, 224)):
        """Получение карты внимания"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.activations = []
        
        with torch.no_grad():
            _ = self.cnn(image_tensor)
        
        if self.activations:
            attention = self.activations[0].cpu().squeeze()
            attention = attention.mean(dim=0).numpy()
            
            # Нормализация
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # Изменение размера
            zoom_factors = (target_size[0] / attention.shape[0], 
                          target_size[1] / attention.shape[1])
            attention = zoom(attention, zoom_factors, order=1)
            
            return attention
        
        return np.zeros(target_size)