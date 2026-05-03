// components/FileUpload.tsx
import React from 'react';
import { useFileUpload } from '../hooks/useFileUpload';

interface FileUploadProps {
  onFileSelected: (file: File | null) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelected }) => {
  const {
    selectedFile,
    isDragging,
    handleFile,
    removeFile,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    formatFileSize
  } = useFileUpload();

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const files = e.target.files;
    if (files && files.length > 0) {
      try {
        handleFile(files[0]);
        onFileSelected(files[0]);
      } catch (error) {
        console.error('File handling error:', error);
      }
    }
  };

  const handleRemove = (): void => {
    removeFile();
    onFileSelected(null);
  };

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    try {
      handleDrop(e);
      const file = e.dataTransfer.files[0];
      if (file) {
        onFileSelected(file);
      }
    } catch (error) {
      console.error('File drop error:', error);
    }
  };

  return (
    <div className="card upload-section">
      <div className="card-header">
        <h2>📄 Submit Insurance Document</h2>
      </div>
      <div className="card-body">
        {!selectedFile ? (
          <div
            className={`upload-area ${isDragging ? 'dragover' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleFileDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
          >
            <div className="upload-icon">📁</div>
            <div className="upload-text">Drop your ACORD PDF here</div>
            <div className="upload-subtext">or click to browse files</div>
            <input
              type="file"
              id="fileInput"
              accept=".pdf"
              hidden
              onChange={handleFileInput}
            />
          </div>
        ) : (
          <div className="file-info show">
            <span>📎</span>
            <span>{selectedFile.name}</span>
            <span style={{ color: 'var(--text-light)' }}>
              {formatFileSize(selectedFile.size)}
            </span>
            <button className="remove-btn" onClick={handleRemove}>
              ✕ Remove
            </button>
          </div>
        )}
      </div>
    </div>
  );
};