// App.tsx
import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { FileUpload } from './components/FileUpload';
import { ProgressTracker } from './components/ProgressTracker';
import { ReportView } from './components/ReportView';
import { Sidebar } from './components/Sidebar';
import { Toast } from './components/Toast';
import { useSubmission } from './hooks/useSubmission';
import type { ReportData } from './types';
import type { ToastMessage } from './types';
import './styles.css';


function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [showUpload, setShowUpload] = useState(true);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [processingTime, setProcessingTime] = useState<string>('-');

  const {
    isProcessing,
    progress,
    agentStatus,
    submitFile,
    startPolling,
    cancelPolling
  } = useSubmission();

  const addToast = useCallback((message: string, type: ToastMessage['type'] = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  }, []);

  const removeToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const handleFileSelected = useCallback((file: File | null) => {
    setSelectedFile(file);
  }, []);

  const handleSubmit = async () => {
    if (!selectedFile) return;

    try {
      const submissionId = await submitFile(selectedFile);
      addToast('Submission received! Processing...', 'success');

      startPolling(
        submissionId,
        (reportData) => {
          setReport(reportData);
          setShowReport(true);
          setShowUpload(false);
          setProcessingTime(
            reportData.processing_time_seconds 
              ? `${reportData.processing_time_seconds.toFixed(1)}s` 
              : '-'
          );
          addToast('Report generated successfully!', 'success');
        },
        (error) => {
          addToast(`Processing failed: ${error}`, 'error');
        }
      );
    } catch (error: any) {
      addToast(`Error: ${error.message}`, 'error');
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setReport(null);
    setShowReport(false);
    setShowUpload(true);
    setProcessingTime('-');
    cancelPolling();
  };

  return (
    <>
      <Header />
      
      <div className="main-container">
        <div>
          {showUpload && (
            <>
              <FileUpload onFileSelected={handleFileSelected} />
              
              <button
                className="btn btn-primary btn-block"
                disabled={!selectedFile || isProcessing}
                onClick={handleSubmit}
                style={{ marginTop: '20px' }}
              >
                {isProcessing ? (
                  <>
                    <span className="spinner" />
                    Processing...
                  </>
                ) : (
                  '🚀 Process Submission'
                )}
              </button>

              <ProgressTracker
                progress={progress}
                agentStatus={agentStatus}
                isVisible={isProcessing}
              />
            </>
          )}

          <ReportView
            report={report}
            isVisible={showReport}
            onReset={handleReset}
          />
        </div>

        <Sidebar report={report} processingTime={processingTime} />
      </div>

      <div className="toast-container">
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </>
  );
}

export default App;