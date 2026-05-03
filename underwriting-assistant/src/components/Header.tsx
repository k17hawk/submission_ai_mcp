// components/Header.tsx
import React from 'react';
import { useHealthCheck } from '../hooks/useHealthCheck';

export const Header: React.FC = () => {
  const { health, isConnected } = useHealthCheck();

  const mcpStatus = health?.mcp_servers;
  const insuranceOk = mcpStatus?.insurance?.status?.includes('✅');
  const riskOk = mcpStatus?.risk?.status?.includes('✅');
  
  return (
    <div className="header">
      <div className="header-content">
        <div className="logo">
          <div className="logo-icon">🛡️</div>
          <span>Underwriting Assistant</span>
        </div>
        <div className="status-badge">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          <span>{isConnected ? 'Servers Connected' : 'API Disconnected'}</span>
          {isConnected && insuranceOk && riskOk && (
            <span style={{ marginLeft: '8px' }}>
              (2 MCP Servers: {mcpStatus?.insurance?.tools || 0} + {mcpStatus?.risk?.tools || 0} tools)
            </span>
          )}
        </div>
      </div>
    </div>
  );
};