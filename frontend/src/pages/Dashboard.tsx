import { useEffect } from 'react'
import { getAllForecasts, getAgentStatus } from '../services/api'
import { usePortfolioStore } from '../store/portfolioStore'
import StatCard from '../components/StatCard'
import AllocationPie from '../components/AllocationPie'

export default function Dashboard() {
  const {
    forecasts, setForecasts, setForecastsLoading,
    portfolio, agentStatus, setAgentStatus,
  } = usePortfolioStore()

  useEffect(() => {
    setForecastsLoading(true)
    getAllForecasts()
      .then(r => setForecasts(r.data.forecasts))
      .catch(() => {})
      .finally(() => setForecastsLoading(false))

    getAgentStatus().then(r => setAgentStatus(r.data)).catch(() => {})
  }, [])

  const pairs = Object.keys(forecasts)

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Portfolio Value"
          value={portfolio ? `$${portfolio.expected_amount.toLocaleString()}` : '—'}
          sub={portfolio ? `Budget: $${portfolio.budget.toLocaleString()}` : 'Run optimizer first'}
        />
        <StatCard
          label="Expected Return"
          value={portfolio ? `${portfolio.expected_return_pct.toFixed(2)}%` : '—'}
          color="text-green-400"
        />
        <StatCard
          label="Risk Level"
          value={portfolio?.risk_tolerance ?? '—'}
          color="text-yellow-400"
        />
        <StatCard
          label="Pipeline Runs"
          value={agentStatus?.pipeline_runs ?? 0}
          sub={`Agents: ${agentStatus ? 'Active' : 'Idle'}`}
        />
      </div>

      {/* Forecast returns */}
      {pairs.length > 0 && (
        <div>
          <h2 className="text-base font-semibold text-muted mb-3">Live Forecast Returns</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {pairs.map(pair => {
              const f = forecasts[pair]
              const isPositive = f.forecast_return >= 0
              return (
                <div key={pair} className="bg-surface rounded-xl p-4 space-y-1">
                  <p className="text-sm font-semibold">{pair.replace('_', '/')}</p>
                  <p className="text-xs text-muted">Last Actual: {f.last_actual.toFixed(5)}</p>
                  <p className="text-xs text-muted">Predicted:   {f.predicted_next.toFixed(5)}</p>
                  <p className={`text-lg font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                    {isPositive ? '+' : ''}{f.forecast_return_pct}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Portfolio allocation */}
      {portfolio && (
        <div>
          <h2 className="text-base font-semibold text-muted mb-3">Current Allocation</h2>
          <div className="max-w-sm">
            <AllocationPie
              allocations={Object.fromEntries(
                Object.entries(portfolio.allocations).map(([k, v]) => [k, v.weight_pct])
              )}
            />
          </div>
        </div>
      )}
    </div>
  )
}
