export type AgentLog = {
  agent: string
  stage?: string
  status: 'running' | 'success' | 'failed' | 'skipped' | string
  message?: string
  detail?: any
  timestamp?: number
}

