// components/Sidebar.tsx
import React from 'react';
import type { ReportData } from '../types';

interface SidebarProps {
  report: ReportData | null;
  processingTime: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ report, processingTime }) => {
  return (
    <div className="sidebar">
      <div className="card">
        <div className="card-header">
          <h2>📈 Processing Stats</h2>
        </div>
        <div className="card-body">
          <div className="stat-item">
            <span>Agents Used</span>
            <span className="stat-value">5</span>
          </div>
          <div className="stat-item">
            <span>Processing Time</span>
            <span className="stat-value">
              {processingTime || '-'}
            </span>
          </div>
          <div className="stat-item">
            <span>Clauses Analyzed</span>
            <span className="stat-value">
              {report?.clause_analyses?.length || report?.agent_logs?.length || '-'}
            </span>
          </div>
          <div className="stat-item">
            <span>Errors</span>
            <span className="stat-value">
              {report?.errors?.length || 0}
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>🤖 Agent Pipeline</h2>
        </div>
        <div className="card-body">
          <div className="stat-item">
            <span>🔍 Extractor</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>
              Insurance Server
            </span>
          </div>
          <div className="stat-item">
            <span>📊 Analyzer</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>
              Insurance Server
            </span>
          </div>
          <div className="stat-item">
            <span>⚠️ Risk Assessor</span>
            <span style={{ fontSize: '0.75rem', color: '#2563eb' }}>
              Risk Server
            </span>
          </div>
          <div className="stat-item">
            <span>🎯 Advisor</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>
              LangGraph
            </span>
          </div>
          <div className="stat-item">
            <span>📋 Reporter</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>
              LangGraph
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>📚 Recent</h2>
        </div>
        <div className="card-body">
          <p style={{ color: 'var(--text-light)', fontSize: '0.875rem' }}>
            No submissions yet
          </p>
        </div>
      </div>
    </div>
  );
};