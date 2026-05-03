import type { Agent } from '../types';

export const AGENTS: Agent[] = [
  { name: 'Extractor', emoji: '🔍', server: 'Insurance' },
  { name: 'Analyzer', emoji: '📊', server: 'Insurance' },
  { name: 'Risk Assessor', emoji: '⚠️', server: 'Risk' },
  { name: 'Advisor', emoji: '🎯', server: 'Internal' },
  { name: 'Reporter', emoji: '📋', server: 'Internal' },
];
export const API_BASE = import.meta.env.VITE_API_BASE || '';

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
export const ALLOWED_FILE_TYPES = ['.pdf'];
export const POLLING_INTERVAL = 1000; // 1 second
export const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds