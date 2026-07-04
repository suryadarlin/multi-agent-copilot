import { useMemo, useState, type ReactNode } from 'react'
import axios from 'axios'
import {
  ArrowPathIcon,
  BugAntIcon,
  CheckCircleIcon,
  ClipboardDocumentIcon,
  ClipboardDocumentListIcon,
  Cog6ToothIcon,
  PlayIcon,
  ShieldCheckIcon,
  SparklesIcon,
  WrenchIcon,
} from './icons'
import type { AgentLog } from './types'

type StageId =
  | 'Planning'
  | 'Code Generation'
  | 'Critic Review'
  | 'Security Scan'
  | 'Testing'
  | 'Debugging'
  | 'Auto Fix'
  | 'Completed'

type RunResponse = {
  success?: boolean
  stage_reached?: string
  generated_files?: Record<string, string>
  events?: any[]
  critic_feedback?: any
}

const STAGES: { id: StageId; icon: React.ReactNode; color: string }[] = [
  { id: 'Planning', icon: <SparklesIcon />, color: 'bg-indigo-500' },
  { id: 'Code Generation', icon: <WrenchIcon />, color: 'bg-cyan-500' },
  { id: 'Critic Review', icon: <Cog6ToothIcon />, color: 'bg-amber-500' },
  { id: 'Security Scan', icon: <ShieldCheckIcon />, color: 'bg-emerald-500' },
  { id: 'Testing', icon: <CheckCircleIcon />, color: 'bg-green-600' },
  { id: 'Debugging', icon: <BugAntIcon />, color: 'bg-violet-500' },
  { id: 'Auto Fix', icon: <ArrowPathIcon />, color: 'bg-orange-500' },
  { id: 'Completed', icon: <CheckCircleIcon />, color: 'bg-sky-500' },
]

function computeHealthScore(opts: { events: AgentLog[]; success?: boolean }) {
  const { events, success } = opts
  const total = STAGES.length - 1 // exclude Completed
  const stageIds = new Set<string>()

  for (const e of events) {
    const a = (e.agent || '').toLowerCase()
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

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStage, setCurrentStage] = useState<StageId | null>(null)

  const [events, setEvents] = useState<AgentLog[]>([])
  const [generatedFiles, setGeneratedFiles] = useState<Record<string, string> | null>(null)
  const [healthScore, setHealthScore] = useState<number | null>(null)

  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8501'

  const stageOrder = useMemo(() => STAGES.map((s) => s.id), [])

  async function handleRun() {
    if (!prompt.trim()) return

    setIsRunning(true)
    setProgress(0)
    setCurrentStage(null)
    setEvents([])
    setGeneratedFiles(null)
    setHealthScore(null)

    try {
      const resp = await axios.post(`${apiBase}/api/copilot/run`, {
        requirement: prompt,
      })


      const data: RunResponse = resp.data

      // Backend adapter should return structured events/logs.
      const mappedEvents: AgentLog[] = ((data.events || data.logs || []) as any[]).map((e: any) => ({
        agent: e.agent || e.agent_name || 'Agent',
        stage: e.stage,
        status: e.status || 'running',
        message: e.message || e.msg || '',
        detail: e.detail || {},
        timestamp: e.timestamp,
      }))

      setEvents(mappedEvents)
      setGeneratedFiles(data.generated_files || {})
      setHealthScore(computeHealthScore({ events: mappedEvents, success: data.success }))

      let idx = -1
      const stageMap = new Map<string, StageId>([
        ['Planner Agent', 'Planning'],
        ['Code Agent', 'Code Generation'],
        ['Critic Agent', 'Critic Review'],
        ['Test Agent', 'Testing'],
        ['Security Agent', 'Security Scan'],
        ['Debug Agent', 'Debugging'],
        ['Auto Fix Agent', 'Auto Fix'],
      ])

      for (const ev of mappedEvents) {
        const mapped = stageMap.get(ev.agent)
        if (!mapped) continue
        idx = Math.max(idx, stageOrder.indexOf(mapped))
        setCurrentStage(mapped)
        setProgress((idx + 1) / stageOrder.length)
      }

      setCurrentStage('Completed')
      setProgress(1)
    } catch (err: any) {
      const msg = err?.response?.data?.message || err?.message || 'Failed to start workflow'
      setEvents([
        {
          agent: 'Frontend',
          status: 'failed',
          message:
            `${msg}\n\nExpected backend endpoint: POST /api/copilot/run and GET /api/copilot/zip.`,
          detail: {},
          timestamp: Date.now(),
          stage: 'Completed',
        },
      ])
      setHealthScore(0)
      setProgress(0)
      setCurrentStage('Completed')
    } finally {
      setIsRunning(false)
    }
  }

  function downloadZip() {
    const apiBase2 = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8501'
    window.open(
      `${apiBase2}/api/copilot/zip?requirement=${encodeURIComponent(prompt)}`,
      '_blank',
    )
  }

  const stageProgress = Math.round(progress * 100)

  return (
    <div className="min-h-screen">
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="text-3xl font-semibold">Multi-Agent AI Engineering Copilot</h1>
            <p className="mt-2 text-slate-300">
              Submit a single requirement and monitor: Planning → Code → Critic → Security → Testing → Debugging → Auto Fix.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <div className="text-slate-400 text-sm">Project Health Score</div>
            <div className="mt-1 text-2xl font-bold">{healthScore ?? '—'}</div>
            <div className="text-slate-400 text-sm">/ 100</div>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm text-slate-300" htmlFor="prompt">
              Requirement
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-[120px] w-full resize-none rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder='e.g. "Build a FastAPI JWT auth system with refresh tokens"'
            />

            <div className="flex items-center justify-between gap-4">
              <button
                onClick={handleRun}
                disabled={isRunning || !prompt.trim()}
                className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                <PlayIcon />
                {isRunning ? 'Running…' : 'Generate'}
              </button>

              <div className="flex items-center gap-3 text-slate-300">
                <div className="text-sm">Progress: {stageProgress}%</div>
                {currentStage ? (
                  <div className="text-sm">
                    Stage: <span className="font-medium text-white">{currentStage}</span>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="mt-3">
              <div className="h-2 w-full rounded-full bg-slate-800">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-[width]"
                  style={{ width: `${progress * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
              <div className="flex items-center justify-between">
                <div className="text-lg font-semibold">Execution Monitor</div>
                <div className="text-sm text-slate-400">Live stage progress</div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                {STAGES.map((s) => {
                  const idx = stageOrder.indexOf(s.id)
                  const done = progress >= (idx + 1) / stageOrder.length || s.id === 'Completed'
                  const active = currentStage === s.id
                  return (
                    <div
                      key={s.id}
                      className={`rounded-xl border p-3 ${
                        active
                          ? 'border-indigo-400 bg-indigo-500/10'
                          : done
                            ? 'border-emerald-500/40 bg-emerald-500/10'
                            : 'border-slate-800 bg-slate-950/20'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`inline-flex items-center justify-center rounded-lg ${s.color} p-2`}>
                          {s.icon}
                        </div>
                        <div>
                          <div className="font-medium">{s.id}</div>
                          <div className="text-xs text-slate-400">
                            {active ? 'running' : done ? 'done' : 'pending'}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
              <div className="flex items-center justify-between">
                <div className="text-lg font-semibold">Agent Logs</div>
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <ClipboardDocumentListIcon />
                </div>
              </div>

              <div className="mt-3 max-h-[420px] overflow-auto rounded-xl border border-slate-800 bg-slate-950">
                {events.length === 0 ? (
                  <div className="p-4 text-slate-400">Run the pipeline to see logs from every agent.</div>
                ) : (
                  <div className="space-y-3 p-3">
                    {events.map((e, i) => {
                      const status = String(e.status).toLowerCase()
                      const badge =
                        status === 'failed'
                          ? 'bg-red-500/20 text-red-200 border-red-500/40'
                          : status === 'success'
                            ? 'bg-emerald-500/20 text-emerald-200 border-emerald-500/40'
                            : status === 'skipped'
                              ? 'bg-slate-600/20 text-slate-300 border-slate-500/40'
                              : 'bg-indigo-500/20 text-indigo-200 border-indigo-500/40'

                      return (
                        <div key={i} className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <div className="font-medium">{e.agent}</div>
                                <span
                                  className={`rounded-full border px-2 py-0.5 text-xs ${badge}`}
                                >
                                  {e.status}
                                </span>
                              </div>
                              {e.message ? (
                                <div className="mt-2 whitespace-pre-wrap text-sm text-slate-200">
                                  {e.message}
                                </div>
                              ) : null}
                            </div>
                          </div>

                          {e.detail && Object.keys(e.detail).length ? (
                            <pre className="mt-2 max-h-28 overflow-auto rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-300">
                              {JSON.stringify(e.detail, null, 2)}
                            </pre>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-1 space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
              <div className="text-lg font-semibold">Generated Files</div>
              <div className="mt-2 text-sm text-slate-400">List returned by the backend pipeline</div>

              <div className="mt-3 max-h-[220px] overflow-auto rounded-xl border border-slate-800 bg-slate-950">
                {generatedFiles && Object.keys(generatedFiles).length ? (
                  <ul className="p-3 space-y-2">
                    {Object.keys(generatedFiles).map((k) => (
                      <li key={k} className="break-all text-sm text-slate-200">• {k}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="p-4 text-slate-400">No files yet.</div>
                )}
              </div>

              <div className="mt-4">
                <button
                  onClick={downloadZip}
                  disabled={!generatedFiles}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-slate-800 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <ClipboardDocumentIcon />
                  Download generated project
                </button>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
              <div className="text-lg font-semibold">Integration Note</div>
              <div className="mt-2 text-sm text-slate-300">
                This UI expects a REST adapter that runs the existing backend pipeline and returns events + artifacts.
              </div>
              <ul className="mt-3 list-disc pl-5 text-sm text-slate-200">
                <li>
                  <code className="text-xs">POST /api/copilot/run</code>
                </li>
                <li>
                  <code className="text-xs">GET /api/copilot/zip</code>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

