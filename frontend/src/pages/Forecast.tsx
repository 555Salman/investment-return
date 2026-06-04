import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { getAllForecasts, getPairHistory } from '../services/api'
import { usePortfolioStore } from '../store/portfolioStore'

const PAIRS = ['EUR_USD', 'AUD_USD', 'NZD_USD']

interface HistoryPoint { index: number; actual: number; predicted: number }

export default function Forecast() {
  const { forecasts, setForecasts, forecastsLoading, setForecastsLoading } = usePortfolioStore()

  // Full test-set series per pair
  const [history, setHistory] = useState<Record<string, HistoryPoint[]>>({})
  const [histLoading, setHistLoading] = useState(false)
  const [error, setError] = useState('')

  const loadForecasts = () => {
    setForecastsLoading(true)
    setError('')
    getAllForecasts()
      .then(r => setForecasts(r.data.forecasts))
      .catch(() => setError('Models not trained yet. Run training first.'))
      .finally(() => setForecastsLoading(false))
  }

  const loadHistory = () => {
    setHistLoading(true)
    Promise.all(
      PAIRS.map(pair =>
        getPairHistory(pair)
          .then(r => ({ pair, data: r.data }))
          .catch(() => null)
      )
    ).then(results => {
      const map: Record<string, HistoryPoint[]> = {}
      results.forEach(res => {
        if (!res) return
        map[res.pair] = res.data.actual.map((a, i) => ({
          index:     i + 1,
          actual:    a,
          predicted: res.data.predicted[i],
        }))
      })
      setHistory(map)
    }).finally(() => setHistLoading(false))
  }

  useEffect(() => {
    loadForecasts()
    loadHistory()
  }, [])

  const loading = forecastsLoading || histLoading

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Exchange Rate Forecasts</h1>
        <button
          onClick={() => { loadForecasts(); loadHistory() }}
          disabled={loading}
          className="bg-accent hover:bg-blue-400 text-white text-sm rounded-lg px-4 py-2 transition-colors disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <div className="grid grid-cols-1 gap-8">
        {PAIRS.map(pair => {
          const f    = forecasts[pair]
          const data = history[pair] ?? []
          const isPos = f ? f.forecast_return >= 0 : true

          return (
            <div key={pair} className="bg-surface rounded-xl p-5 space-y-4">

              {/* Title row */}
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">{pair.replace('_', '/')} — Actual vs Predicted (Test Set)</h2>
                {f && (
                  <span className={`text-sm font-bold px-3 py-1 rounded-full
                    ${isPos ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                    {isPos ? '+' : ''}{f.forecast_return_pct}
                  </span>
                )}
              </div>

              {/* Chart */}
              {data.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="index"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      label={{ value: 'Test Day', position: 'insideBottomRight', fill: '#64748b', fontSize: 11, offset: -4 }}
                    />
                    <YAxis
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      domain={['auto', 'auto']}
                      tickFormatter={(v: number) => v.toFixed(4)}
                      width={72}
                    />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                      labelStyle={{ color: '#94a3b8', fontSize: 11 }}
                      formatter={(v: number, name: string) => [v.toFixed(6), name]}
                      labelFormatter={(l: number) => `Day ${l}`}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      stroke="#0ea5e9"
                      dot={false}
                      strokeWidth={1.8}
                      name="Actual"
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      stroke="#f59e0b"
                      dot={false}
                      strokeWidth={1.8}
                      strokeDasharray="5 3"
                      name="Predicted"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[280px] flex items-center justify-center text-slate-500 text-sm">
                  {histLoading ? 'Loading chart data…' : 'No data available. Train the model first.'}
                </div>
              )}

              {/* Stats row */}
              {f && (
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-400">Last Actual</p>
                    <p className="font-mono text-sm font-semibold">{f.last_actual.toFixed(5)}</p>
                  </div>
                  <div className="bg-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-400">Predicted Next</p>
                    <p className="font-mono text-sm font-semibold">{f.predicted_next.toFixed(5)}</p>
                  </div>
                  <div className="bg-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-400">Forecast Return</p>
                    <p className={`text-sm font-bold ${isPos ? 'text-green-400' : 'text-red-400'}`}>
                      {isPos ? '+' : ''}{f.forecast_return_pct}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
