import { useState } from 'react'
import { optimizePortfolio, getBenchmark } from '../services/api'
import type { BenchmarkRow } from '../services/api'
import { usePortfolioStore } from '../store/portfolioStore'
import AllocationPie from '../components/AllocationPie'

export default function Portfolio() {
  const { portfolio, setPortfolio, portfolioLoading, setPortfolioLoading } = usePortfolioStore()

  const [budget, setBudget]         = useState(10000)
  const [risk, setRisk]             = useState<'low' | 'medium' | 'high'>('medium')
  const [days, setDays]             = useState(365)
  const [benchmarks, setBenchmarks] = useState<BenchmarkRow[]>([])
  const [error, setError]           = useState('')

  const handleOptimize = async () => {
    setPortfolioLoading(true)
    setError('')
    try {
      const [optRes, benchRes] = await Promise.all([
        optimizePortfolio({ budget, risk_tolerance: risk, investment_days: days }),
        getBenchmark({ budget, investment_days: days, risk_tolerance: risk }),
      ])
      setPortfolio(optRes.data)
      setBenchmarks(benchRes.data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Optimization failed. Ensure models are trained.')
    } finally {
      setPortfolioLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Portfolio Optimizer</h1>

      {/* Controls */}
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
          <label className="text-xs text-muted">Risk Tolerance</label>
          <select
            value={risk}
            onChange={e => setRisk(e.target.value as 'low' | 'medium' | 'high')}
            className="bg-slate-800 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-xs text-muted">Investment Days</label>
          <input
            type="number"
            value={days}
            onChange={e => setDays(Number(e.target.value))}
            className="bg-slate-800 rounded-lg px-3 py-2 text-sm w-24 outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
        <button
          onClick={handleOptimize}
          disabled={portfolioLoading}
          className="bg-accent hover:bg-blue-400 text-white rounded-lg px-5 py-2 text-sm font-semibold transition-colors disabled:opacity-50"
        >
          {portfolioLoading ? 'Optimizing…' : 'Optimize'}
        </button>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {portfolio && (
        <>
          {/* Result cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(portfolio.allocations).map(([pair, alloc]) => (
              <div key={pair} className="bg-surface rounded-xl p-4 space-y-1">
                <p className="text-xs text-muted">{pair.replace('_', '/')}</p>
                <p className="text-xl font-bold">{alloc.weight_pct.toFixed(1)}%</p>
                <p className="text-xs text-muted">${alloc.amount_usd.toFixed(2)}</p>
              </div>
            ))}
            <div className="bg-surface rounded-xl p-4 space-y-1">
              <p className="text-xs text-muted">Expected Return</p>
              <p className="text-xl font-bold text-green-400">{portfolio.expected_return_pct.toFixed(2)}%</p>
              <p className="text-xs text-muted">${portfolio.expected_amount.toFixed(2)}</p>
            </div>
          </div>

          {/* Allocation chart */}
          <div className="max-w-sm">
            <AllocationPie
              title="LP Optimised Allocation"
              allocations={Object.fromEntries(
                Object.entries(portfolio.allocations).map(([k, v]) => [k, v.weight_pct])
              )}
            />
          </div>

          {/* Benchmark table */}
          {benchmarks.length > 0 && (
            <div className="bg-surface rounded-xl p-4 overflow-x-auto">
              <h3 className="text-sm font-semibold text-muted mb-3">Strategy Comparison</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-muted border-b border-slate-700">
                    <th className="text-left pb-2">Strategy</th>
                    <th className="text-right pb-2">Return %</th>
                    <th className="text-right pb-2">Amount</th>
                    {Object.keys(benchmarks[0].allocations).map(p => (
                      <th key={p} className="text-right pb-2">{p.replace('_', '/')}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {benchmarks.map(row => (
                    <tr key={row.strategy} className="border-b border-slate-800">
                      <td className="py-2 font-medium capitalize">{row.strategy.replace('_', ' ')}</td>
                      <td className="text-right text-green-400">{row.expected_return_pct.toFixed(2)}%</td>
                      <td className="text-right">${row.expected_amount.toFixed(2)}</td>
                      {Object.values(row.allocations).map((v, i) => (
                        <td key={i} className="text-right">{v.toFixed(1)}%</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
