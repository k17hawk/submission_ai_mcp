// Agent types
export interface Agent {
  name: string;
  emoji: string;
  server: 'Insurance' | 'Risk' | 'Internal';
}

export type AgentStatus = 'pending' | 'active' | 'completed' | 'error';

// Health check types
export interface MCPTool {
  name: string;
  description?: string;
}

export interface MCPServerStatus {
  status: string;
  tools: number;
  available_tools?: MCPTool[];
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  mcp_servers?: {
    insurance?: MCPServerStatus;
    risk?: MCPServerStatus;
  };
  langgraph?: {
    status: string;
    agents: number;
  };
}

// Submission types
export interface SubmissionResponse {
  submission_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  error?: string;
  message?: string;
}

// Report types
export interface RiskFactor {
  name: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface ClauseAnalysis {
  clause_name: string;
  rating: number;
  comments: string;
}

export interface AgentLog {
  agent: string;
  status: 'success' | 'error';
  message: string;
  timestamp: string;
}

export interface ReportData {
  submission_id?: string;
  policy_type?: string;
  insured_name?: string;
  policy_number?: string;
  final_decision?: string;
  decision_emoji?: string;
  average_rating?: number;
  full_report?: string;
  executive_summary?: string;
  risk_factors?: RiskFactor[];
  strong_points?: string[];
  required_actions?: string[];
  clause_analyses?: ClauseAnalysis[];
  agent_logs?: AgentLog[];
  errors?: string[];
  processing_time_seconds?: number;
  timestamp?: string;
}

// Toast types
export interface ToastMessage {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

// File upload types
export interface FileUploadState {
  file: File | null;
  isDragging: boolean;
  error: string | null;
}