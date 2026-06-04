import type { AgentLogEntry } from '../services/api'

interface Props { entries: AgentLogEntry[] }

const AGENT_COLORS: Record<string, string> = {
  MarketMonitor:  'text-yellow-400',
  ForecastAgent:  'text-blue-400',
  DecisionAgent:  'text-purple-400',
  RebalanceAgent: 'text-green-400',
}

export default function AgentLog({ entries }: Props) {
  return (
    <div className="bg-surface rounded-xl p-4 h-80 overflow-y-auto font-mono text-xs">
      {entries.length === 0 && (
        <p className="text-muted text-center mt-8">No agent activity yet. Run the pipeline to start.</p>
      )}
      {[...entries].reverse().map((e, i) => (
        <div key={i} className="mb-2 border-b border-slate-700 pb-2">
          <span className="text-slate-500">{e.timestamp.replace('T', ' ').slice(0, 19)}</span>
          {' '}
          <span className={AGENT_COLORS[e.agent] ?? 'text-white'}>[{e.agent}]</span>
          {' '}
          <span className="text-white">{e.action}</span>
          {Object.keys(e.detail).length > 0 && (
            <span className="text-slate-400"> — {JSON.stringify(e.detail)}</span>
          )}
        </div>
      ))}
    </div>
  )
}
