export type HealthStage =
  | 'Planning'
  | 'Code Generation'
  | 'Critic Review'
  | 'Security Scan'
  | 'Testing'
  | 'Debugging'
  | 'Auto Fix'
  | 'Completed'

export type AgentLog = {
  agent: string
  stage?: string
  status: 'running' | 'success' | 'failed' | 'skipped' | string
  message?: string
  detail?: any
  timestamp?: number
}

const STAGES: HealthStage[] = [
  'Planning',
  'Code Generation',
  'Critic Review',
  'Security Scan',
  'Testing',
  'Debugging',
  'Auto Fix',
  'Completed',
]

export function computeHealthScore(opts: {
  events: AgentLog[]
  success?: boolean
}) {
  const { events, success } = opts
  const total = STAGES.length - 1 // exclude Completed
  const stageIds = new Set<string>()

  for (const e of events) {
    if (!e.agent) continue
    const a = e.agent.toLowerCase()
    if (a.includes('planner')) stageIds.add('Planning')
    else if (a.includes('code')) stageIds.add('Code Generation')
    else if (a.includes('critic')) stageIds.add('Critic Review')
    else if (a.includes('security')) stageIds.add('Security Scan')
    else if (a.includes('test')) stageIds.add('Testing')
    else if (a.includes('debug')) stageIds.add('Debugging')
    else if (a.includes('auto')) stageIds.add('Auto Fix')
  }

  const coverageScore = Math.round((stageIds.size / total) * 70)
  const failurePenalty = events.some((e) => String(e.status).toLowerCase() === 'failed')
    ? 35
    : 0
  const successBonus = success ? 30 : 0

  return Math.max(0, Math.min(100, coverageScore - failurePenalty + successBonus))
}

