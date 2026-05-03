import { useState, useCallback, useRef, useEffect } from 'react';
import type { AgentStatus, ReportData } from '../types';
import { AGENTS, POLLING_INTERVAL, API_BASE } from '../constants';

export function useSubmission() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [agentStatus, setAgentStatus] = useState<AgentStatus[]>(
    AGENTS.map(() => 'pending')
  );
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const submitFile = useCallback(async (file: File): Promise<string> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/submit`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    console.log('Submit response data:', data);

    if (!response.ok) {
      const errorMessage = data.detail || data.error || data.message || 'Submission failed';
      throw new Error(errorMessage);
    }

    // Handle CallToolResult or any nested response
    let submissionId = data.submission_id;
    
    if (!submissionId && data.result) {
      // If result is a stringified JSON, parse it
      if (typeof data.result === 'string') {
        try {
          const parsed = JSON.parse(data.result);
          submissionId = parsed.submission_id;
        } catch (e) {
          console.error('Failed to parse result:', e);
        }
      } else if (typeof data.result === 'object') {
        submissionId = data.result.submission_id;
      }
    }

    if (!submissionId) {
      throw new Error('No submission ID received');
    }

    return submissionId;
  }, []);

  const startPolling = useCallback((
    submissionId: string,
    onComplete: (report: ReportData) => void,
    onError: (error: string) => void
  ) => {
    setIsProcessing(true);
    setProgress(10);
    setAgentStatus(AGENTS.map(() => 'pending'));
    
    let agentIndex = 0;

    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/status/${submissionId}`);
        const data = await response.json();
        console.log('Poll response:', data);
        
        // Handle nested response
        let statusData = data;
        if (data.result) {
          statusData = typeof data.result === 'string' 
            ? JSON.parse(data.result) 
            : data.result;
        }

        const status = statusData.status || data.status;
        
        const progressMap: Record<string, number> = {
          'queued': 10,
          'processing': 30 + (agentIndex * 15),
          'completed': 100,
          'failed': 100,
        };
        setProgress(progressMap[status] || 30);

        if (status === 'processing' && agentIndex < AGENTS.length) {
          setAgentStatus(prev => {
            const updated: AgentStatus[] = [...prev];
            if (agentIndex < updated.length) {
              updated[agentIndex] = 'active';
            }
            return updated;
          });

          setTimeout(() => {
            setAgentStatus(prev => {
              const updated: AgentStatus[] = [...prev];
              if (agentIndex < updated.length) {
                updated[agentIndex] = 'completed';
              }
              return updated;
            });
            agentIndex++;
          }, 2000);
        }

        if (status === 'completed') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          setAgentStatus(AGENTS.map(() => 'completed' as AgentStatus));
          setProgress(100);

          try {
            const reportResponse = await fetch(`${API_BASE}/api/report/${submissionId}`);
            const reportData = await reportResponse.json();
            console.log('Report data:', reportData);
            
            // Handle nested response
            let report: ReportData;
            if (reportData.result) {
              report = typeof reportData.result === 'string'
                ? JSON.parse(reportData.result)
                : reportData.result;
            } else {
              report = reportData;
            }
            
            setIsProcessing(false);
            onComplete(report);
          } catch (err) {
            setIsProcessing(false);
            onError(err instanceof Error ? err.message : 'Failed to fetch report');
          }
        } else if (status === 'failed') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          if (agentIndex < AGENTS.length) {
            setAgentStatus(prev => {
              const updated: AgentStatus[] = [...prev];
              updated[agentIndex] = 'error';
              return updated;
            });
          }

          setIsProcessing(false);
          onError(statusData.error || data.error || 'Unknown error');
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    };

    poll();
    pollingIntervalRef.current = setInterval(poll, POLLING_INTERVAL);
  }, []);

  const cancelPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setIsProcessing(false);
    setProgress(0);
    setAgentStatus(AGENTS.map(() => 'pending' as AgentStatus));
  }, []);

  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  return {
    isProcessing,
    progress,
    agentStatus,
    submitFile,
    startPolling,
    cancelPolling,
  };
}