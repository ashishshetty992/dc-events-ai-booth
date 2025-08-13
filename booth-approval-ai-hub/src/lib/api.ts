/**
 * API Configuration for different environments
 * Handles dynamic API endpoint resolution for development and production
 */

// Environment-aware API base URL
const getApiBaseUrl = (): string => {
  // Check for Railway environment variable
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Environment-based detection
  if (import.meta.env.PROD) {
    // Try Railway internal networking first, then fallback
    return import.meta.env.VITE_RAILWAY_BACKEND_URL || 
           'https://gentle-numbers-happen.loca.lt';
  } else {
    return 'http://localhost:8000';  // Development
  }
};

const API_BASE_URL = getApiBaseUrl();

export const API_ENDPOINTS = {
  // Core endpoints
  SUBMIT_BOOTH_REQUEST: `${API_BASE_URL}/submit_booth_request`,
  CHAT_HISTORY: `${API_BASE_URL}/chat_history`,
  CHAT_HISTORY_BY_ID: (id: string) => `${API_BASE_URL}/chat_history/${id}`,
  
  // Approval endpoints
  APPROVAL_SAVE: `${API_BASE_URL}/api/approval/save`,
  APPROVAL_BY_ID: (id: string) => `${API_BASE_URL}/api/approval/${id}`,
  APPROVALS_ALL: `${API_BASE_URL}/api/approvals/all`,
  
  // Booth design endpoints
  BOOTH_DESIGN_CHAT: `${API_BASE_URL}/api/booth-design/chat`,
  BOOTH_DESIGN_LOAD: (id: string) => `${API_BASE_URL}/api/booth-design/load/${id}`,
  BOOTH_DESIGN_SAVE: `${API_BASE_URL}/api/booth-design/save`,
  FINALIZE_DESIGN: `${API_BASE_URL}/api/finalize-design`,
  
  // Chat endpoints
  CHAT_SAVE: `${API_BASE_URL}/api/chat/save`,
  
  // Analytics endpoints
  ANALYTICS: `${API_BASE_URL}/analytics`,
  ANALYTICS_BOOTH_PERFORMANCE: `${API_BASE_URL}/analytics/booth_performance`,
  ANALYTICS_VENDOR_INSIGHTS: `${API_BASE_URL}/analytics/vendor_insights`,
  ANALYTICS_MARKET_TRENDS: `${API_BASE_URL}/analytics/market_trends`,
  ANALYTICS_COMPLIANCE_REPORTS: `${API_BASE_URL}/analytics/compliance_reports`,
  
  // Health check
  HEALTH: `${API_BASE_URL}/api/health`,
} as const;

/**
 * Get the full API URL for a given endpoint
 */
export const getApiUrl = (endpoint: string): string => `${API_BASE_URL}${endpoint}`;

/**
 * WebSocket URL for real-time communication
 */
export const getWebSocketUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = import.meta.env.PROD ? 'gentle-numbers-happen.loca.lt' : 'localhost:8000';
  return `${protocol}//${host}/ws`;
};

/**
 * Environment detection
 */
export const isDevelopment = !import.meta.env.PROD;
export const isProduction = import.meta.env.PROD;
