import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import AttentionMap from './components/AttentionMap';
import toast, { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setImagePreview(URL.createObjectURL(file));
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      
      if (result.success) {
        setPrediction(result.data);
        toast.success('Analysis completed!');
      } else {
        toast.error('Analysis failed');
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Toaster position="top-right" />
      
      <header className="header">
        <motion.h1
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          🫁 Pneumonia Detection AI
        </motion.h1>
        <p>Upload chest X-ray image for AI-powered analysis</p>
      </header>
      
      <div className="container">
        <div className="upload-section">
          <ImageUploader onImageUpload={handleImageUpload} loading={loading} />
        </div>
        
        {imagePreview && !loading && prediction && (
          <motion.div
            className="results-section"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <ResultDisplay prediction={prediction} />
            <AttentionMap 
              attentionMap={prediction.attention_map}
              originalImage={imagePreview}
            />
          </motion.div>
        )}
        
        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Analyzing image...</p>
          </div>
        )}
      </div>
      
      <footer className="footer">
        <p>Powered by ResNet18 + Genetic Algorithm | Clinical decision support tool</p>
      </footer>
    </div>
  );
}

export default App;