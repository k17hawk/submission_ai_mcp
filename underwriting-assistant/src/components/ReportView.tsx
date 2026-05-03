// components/ReportView.tsx
import React from 'react';
import  type { ReportData } from '../types';


interface ReportViewProps {
  report: ReportData | null;
  isVisible: boolean;
  onReset: () => void;
}

export const ReportView: React.FC<ReportViewProps> = ({ report, isVisible, onReset }) => {
  if (!isVisible || !report) return null;

  const getDecisionClass = (decision: string = '') => {
    if (decision.includes('ACCEPT')) return 'decision-accept';
    if (decision.includes('CONDITIONAL')) return 'decision-conditional';
    if (decision.includes('REJECT')) return 'decision-reject';
    return 'decision-refer';
  };

  const getRiskLevel = (rating: number) => {
    if (rating >= 4) return { label: 'LOW RISK', color: '#059669', class: 'risk-low' };
    if (rating >= 2.5) return { label: 'MEDIUM RISK', color: '#d97706', class: 'risk-medium' };
    return { label: 'HIGH RISK', color: '#dc2626', class: 'risk-high' };
  };

  const rating = report.average_rating || 0;
  const risk = getRiskLevel(rating);
  const policyInfo = `${report.policy_type || 'Unknown'} | ${report.insured_name || 'Unknown'} | ${report.policy_number || 'N/A'}`;

  return (
    <div className="report-container show">
      <div className="card">
        <div className="report-header">
          <h2 style={{ fontSize: '1.5rem', marginBottom: '4px' }}>
            📋 Underwriting Report
          </h2>
          <p style={{ opacity: 0.8 }}>{policyInfo}</p>
          <div className={`decision-badge ${getDecisionClass(report.final_decision)}`}>
            {report.decision_emoji} {report.final_decision || 'N/A'}
          </div>
        </div>
        <div className="card-body">
          <h3 style={{ marginBottom: '16px' }}>📊 Risk Assessment</h3>
          <div className="risk-meter">
            <span style={{ fontWeight: 700, fontSize: '1.5rem' }}>
              {rating.toFixed(1)}
            </span>
            <span style={{ color: 'var(--text-light)' }}>/ 5.0</span>
            <div className="risk-meter-bar">
              <div
                className={`risk-meter-fill ${risk.class}`}
                style={{ width: `${(rating / 5) * 100}%` }}
              />
            </div>
            <span style={{ fontWeight: 600, color: risk.color }}>
              {risk.label}
            </span>
          </div>

          <h3 style={{ margin: '24px 0 16px' }}>📝 Full Report</h3>
          <pre
            style={{
              background: '#f8fafc',
              padding: '20px',
              borderRadius: '8px',
              overflowX: 'auto',
              fontSize: '0.875rem',
              lineHeight: 1.8
            }}
          >
            {report.full_report || report.executive_summary || 'No report generated'}
          </pre>

          <div style={{ marginTop: '20px' }}>
            <button className="btn btn-primary" onClick={() => window.print()}>
              🖨️ Print Report
            </button>
            <button
              className="btn"
              onClick={onReset}
              style={{ marginLeft: '10px', background: '#e2e8f0' }}
            >
              🔄 New Submission
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};