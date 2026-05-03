import { useState, useEffect, useCallback } from 'react';
import type { HealthStatus } from '../types';
import { API_BASE, HEALTH_CHECK_INTERVAL } from '../constants';

export function useHealthCheck() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      const data = await response.json();
      console.log('Health check response:', data);
      
      // Handle CallToolResult or nested response
      let healthData = data;
      if (data.result) {
        healthData = typeof data.result === 'string' 
          ? JSON.parse(data.result) 
          : data.result;
      }
      
      setHealth(healthData);
      setIsConnected(healthData.status === 'healthy');
      setError(null);
    } catch (err) {
      console.error('Health check error:', err);
      setIsConnected(false);
      setHealth(null);
      setError(err instanceof Error ? err.message : 'Health check failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return { health, isConnected, isLoading, error, checkHealth };
}