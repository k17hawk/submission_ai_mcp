// components/ProgressTracker.tsx
import React from 'react';
import { AGENTS } from '../constants';

interface ProgressTrackerProps {
  progress: number;
  agentStatus: string[];
  isVisible: boolean;
}

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  progress,
  agentStatus,
  isVisible
}) => {
  if (!isVisible) return null;

  return (
    <div className="progress-container show">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="agent-logs">
        {AGENTS.map((agent, index) => {
          const status = agentStatus[index] || 'pending';
          const statusEmoji = 
            status === 'completed' ? '✅' :
            status === 'active' ? '⏳' :
            status === 'error' ? '❌' : '○';
          
          return (
            <div
              key={agent.name}
              className={`agent-log ${status === 'active' ? 'active' : ''} ${
                status === 'completed' ? 'completed' : ''
              } ${status === 'error' ? 'error' : ''}`}
            >
              <span>{statusEmoji}</span>
              <span>{agent.emoji} {agent.name}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-light)', marginLeft: 'auto' }}>
                {agent.server}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};