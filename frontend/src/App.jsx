import React, { useState } from 'react'
import './App.css'

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)

  async function onGenerate() {
    if (!prompt.trim()) return

    setIsLoading(true)
    setError(null)
    setData(null)

    try {
      const resp = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })

      const json = await resp.json().catch(() => ({}))

      if (!resp.ok) {
        throw new Error(json?.detail || json?.message || `Request failed (${resp.status})`)
      }

      setData(json)
    } catch (e) {
      setError(e?.message || String(e))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <div className="title">AI Engineering Copilot</div>
      </div>

      <div className="card">
        <div className="k">Software request</div>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          type="text"
          style={{ width: '100%', padding: 12, borderRadius: 10, border: '1px solid rgba(255,255,255,0.14)', background: 'rgba(0,0,0,0.25)', color: '#e6edf3', outline: 'none' }}
          placeholder='Create student management API'
        />

        <div style={{ marginTop: 12 }}>
          <button onClick={onGenerate} disabled={isLoading || !prompt.trim()}>
            {isLoading ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <span className="spinner" /> Loading…
              </span>
            ) : (
              'Generate'
            )}
          </button>
        </div>
      </div>

      {error ? <div className="err">{error}</div> : null}

      {data ? (
        <div className="grid">
          <div className="card">
            <div className="k">request_id</div>
            <div className="v">{String(data.request_id)}</div>
          </div>

          <div className="card">
            <div className="k">success</div>
            <div className="v">{String(data.success)}</div>
          </div>

          <div className="card">
            <div className="k">elapsed_ms</div>
            <div className="v">{String(data.elapsed_ms)}</div>
          </div>

          <div className="card">
            <div className="k">generated_files</div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(data.generated_files || []).map((f, i) => (
                <li key={`${f}-${i}`} className="v">
                  {String(f)}
                </li>
              ))}
            </ul>
          </div>

          <div className="card" style={{ gridColumn: '1 / -1' }}>
            <div className="k">critic_feedback</div>
            <pre className="v">{JSON.stringify(data.critic_feedback, null, 2)}</pre>
          </div>
        </div>
      ) : null}
    </div>
  )
}

