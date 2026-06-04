import { useEffect, useState } from 'react'
import { getAgentStatus, getAgentLog, runPipeline } from '../services/api'
import { usePortfolioStore } from '../store/portfolioStore'
import { useWebSocket } from '../hooks/useWebSocket'
import AgentLog from '../components/AgentLog'
import type { AgentLogEntry } from '../services/api'

const STATUS_COLOR: Record<string, string> = {
  idle:    'bg-slate-600',
  running: 'bg-blue-500 animate-pulse',
  alert:   'bg-yellow-500',
  error:   'bg-red-500',
}

export default function Agents() {
  const { agentStatus, setAgentStatus, agentLog, appendAgentLog } = usePortfolioStore()
  const [running, setRunning] = useState(false)
  const [budget, setBudget]   = useState(10000)
  const [risk, setRisk]       = useState('medium')

  useEffect(() => {
    getAgentStatus().then(r => setAgentStatus(r.data)).catch(() => {})
    getAgentLog(100).then(r => appendAgentLog(r.data)).catch(() => {})
  }, [])

  // Live log stream via WebSocket
  useWebSocket('/api/agents/ws/log', (data) => {
    appendAgentLog([data as AgentLogEntry])
  })

  const handleRun = async () => {
    setRunning(true)
    try {
      await runPipeline({ trigger: 'manual', budget, risk_tolerance: risk })
      const [statusRes, logRes] = await Promise.all([
        getAgentStatus(),
        getAgentLog(100),
      ])
      setAgentStatus(statusRes.data)
      appendAgentLog(logRes.data)
    } catch {
      // ignore — errors visible in log
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Agent Monitor</h1>

      {/* Agent status cards */}
      {agentStatus && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {(['MarketMonitor', 'ForecastAgent', 'DecisionAgent', 'RebalanceAgent'] as const).map(name => (
            <div key={name} className="bg-surface rounded-xl p-4 flex items-center gap-3">
              <span className={`w-2.5 h-2.5 rounded-full ${STATUS_COLOR[agentStatus[name]] ?? 'bg-slate-600'}`} />
              <div>
                <p className="text-sm font-medium">{name}</p>
                <p className="text-xs text-muted capitalize">{agentStatus[name]}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pipeline controls */}
      <div className="bg-surface rounded-xl p-5 flex flex-wrap gap-4 items-end">
        <div className="space-y-1">
          <label className="text-xs text-muted">Budget (USD)</label>
          <input
            type="number"
            value={budget}
            onChange={e => setBudget(Number(e.target.value))}
            className="bg-slate-800 rounded-lg px-3 py-2 text-sm w-32 outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-muted">Risk</label>
          <select
            value={risk}
            onChange={e => setRisk(e.target.value)}
            className="bg-slate-800 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <button
          onClick={handleRun}
          disabled={running}
          className="bg-accent hover:bg-blue-400 text-white rounded-lg px-5 py-2 text-sm font-semibold transition-colors disabled:opacity-50"
        >
          {running ? 'Running Pipeline…' : 'Run Pipeline'}
        </button>
        <span className="text-xs text-muted self-center">
          Total runs: {agentStatus?.pipeline_runs ?? 0}
        </span>
      </div>

      {/* Live log */}
      <div>
        <h2 className="text-base font-semibold text-muted mb-3">Agent Decision Log</h2>
        <AgentLog entries={agentLog} />
      </div>
    </div>
  )
}
