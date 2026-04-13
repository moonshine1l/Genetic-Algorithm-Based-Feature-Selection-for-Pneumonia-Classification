import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';

const ImageUploader = ({ onImageUpload, loading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0 && !loading) {
      onImageUpload(acceptedFiles[0]);
    }
  }, [onImageUpload, loading]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.dcm']
    },
    multiple: false,
    disabled: loading
  });
  
  return (
    <motion.div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? 'active' : ''} ${loading ? 'disabled' : ''}`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <input {...getInputProps()} />
      <div className="dropzone-content">
        <div className="upload-icon">📤</div>
        {isDragActive ? (
          <p>Drop the X-ray image here...</p>
        ) : (
          <>
            <p>Drag & drop chest X-ray image here</p>
            <p className="subtext">or click to select file</p>
            <p className="formats">Supported formats: JPG, PNG, DICOM</p>
          </>
        )}
      </div>
    </motion.div>
  );
};

export default ImageUploader;