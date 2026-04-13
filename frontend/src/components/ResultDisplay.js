import React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const ResultDisplay = ({ prediction }) => {
  const isPneumonia = prediction.prediction === 1;
  const confidence = (prediction.confidence * 100).toFixed(1);
  
  const data = [
    { name: 'Normal', value: prediction.probabilities.normal * 100 },
    { name: 'Pneumonia', value: prediction.probabilities.pneumonia * 100 }
  ];
  
  const COLORS = ['#3498db', '#e74c3c'];
  
  return (
    <motion.div className="result-card">
      <div className={`result-header ${isPneumonia ? 'pneumonia' : 'normal'}`}>
        <h2>
          {isPneumonia ? '⚠️ PNEUMONIA DETECTED' : '✅ NORMAL'}
        </h2>
        <p>Confidence: {confidence}%</p>
      </div>
      
      <div className="result-details">
        <div className="metrics">
          <div className="metric-card">
            <h3>Diagnosis</h3>
            <p className={isPneumonia ? 'text-danger' : 'text-success'}>
              {isPneumonia ? 'Pneumonia' : 'Normal'}
            </p>
          </div>
          <div className="metric-card">
            <h3>Confidence Level</h3>
            <div className="confidence-bar">
              <div 
                className={`confidence-fill ${isPneumonia ? 'pneumonia-fill' : 'normal-fill'}`}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <p>{confidence}%</p>
          </div>
        </div>
        
        <div className="chart-container">
          <h3>Probability Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="disclaimer">
        <p>⚠️ This is an AI-assisted diagnostic tool. Always consult with a healthcare professional.</p>
      </div>
    </motion.div>
  );
};

export default ResultDisplay;