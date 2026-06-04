import { useState } from 'react'
import { calculateInvestment } from '../services/api'
import type { InvestmentResponse, PairProjection } from '../services/api'

const PAIRS = ['ALL', 'EUR_USD', 'AUD_USD', 'NZD_USD']
const PAIR_LABELS: Record<string, string> = {
  ALL:     'All Currencies',
  EUR_USD: 'EUR / USD — Euro',
  AUD_USD: 'AUD / USD — Australian Dollar',
  NZD_USD: 'NZD / USD — New Zealand Dollar',
}

const RISK_COLORS: Record<string, string> = {
  low:    'bg-green-900 text-green-300',
  medium: 'bg-yellow-900 text-yellow-300',
  high:   'bg-red-900   text-red-300',
}

function fmt(n: number) {
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function GainBadge({ value }: { value: number }) {
  const pos = value >= 0
  return (
    <span className={`font-semibold ${pos ? 'text-green-400' : 'text-red-400'}`}>
      {pos ? '+' : ''}${fmt(value)}
    </span>
  )
}

function PctBadge({ value }: { value: number }) {
  const pos = value >= 0
  return (
    <span className={`text-sm font-bold ${pos ? 'text-green-400' : 'text-red-400'}`}>
      {pos ? '+' : ''}{value.toFixed(3)}%
    </span>
  )
}

function ProjectionCard({ p }: { p: PairProjection }) {
  return (
    <div className="bg-slate-800 rounded-xl p-5 space-y-4 border border-slate-700">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-base font-bold">{p.pair.replace('_', '/')}</p>
          <p className="text-xs text-slate-400">{p.label}</p>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full font-medium
          ${p.direction === 'up' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
          {p.direction === 'up' ? 'APPRECIATING' : 'DEPRECIATING'}
        </span>
      </div>

      {/* Deposit & weight */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-900 rounded-lg p-3">
          <p className="text-xs text-slate-400">Deposit Amount</p>
          <p className="text-lg font-bold">${fmt(p.deposit_usd)}</p>
          <p className="text-xs text-slate-500">{p.weight_pct.toFixed(1)}% of portfolio</p>
        </div>
        <div className="bg-slate-900 rounded-lg p-3">
          <p className="text-xs text-slate-400">Annual Interest Rate</p>
          <p className="text-lg font-bold text-blue-400">{p.annual_rate_pct.toFixed(2)}%</p>
          <p className="text-xs text-slate-500">Fixed deposit rate</p>
        </div>
      </div>

      {/* Returns breakdown */}
      <div className="space-y-2 text-sm">
        <div className="flex justify-between items-center py-1 border-b border-slate-700">
          <span className="text-slate-400">Interest Earned</span>
          <GainBadge value={p.interest_usd} />
        </div>
        <div className="flex justify-between items-center py-1 border-b border-slate-700">
          <span className="text-slate-400">FX Gain / Loss</span>
          <div className="text-right">
            <GainBadge value={p.fx_gain_loss_usd} />
            <span className="text-xs text-slate-500 ml-2">({p.fx_return_pct >= 0 ? '+' : ''}{p.fx_return_pct.toFixed(3)}%)</span>
          </div>
        </div>
        <div className="flex justify-between items-center py-1">
          <span className="text-slate-300 font-medium">Total Gain</span>
          <GainBadge value={p.total_gain_usd} />
        </div>
      </div>

      {/* Projected value */}
      <div className="bg-slate-900 rounded-lg p-3 flex justify-between items-center">
        <span className="text-slate-400 text-sm">Projected Value</span>
        <div className="text-right">
          <p className="text-lg font-bold">${fmt(p.projected_usd)}</p>
          <PctBadge value={p.total_return_pct} />
        </div>
      </div>
    </div>
  )
}

export default function Investment() {
  const [budget, setBudget]     = useState<string>('10000000')
  const [years, setYears]       = useState<string>('2')
  const [risk, setRisk]         = useState<'low' | 'medium' | 'high'>('medium')
  const [pairFilter, setPair]   = useState('ALL')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const [result, setResult]     = useState<InvestmentResponse | null>(null)

  const handleCalculate = async () => {
    const b = parseFloat(budget.replace(/,/g, ''))
    const y = parseFloat(years)
    if (!b || b <= 0 || !y || y <= 0) {
      setError('Please enter a valid budget and number of years.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await calculateInvestment({
        budget: b, years: y, risk_tolerance: risk, pair_filter: pairFilter,
      })
      setResult(res.data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Calculation failed. Make sure models are trained.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Investment Calculator</h1>
        <p className="text-sm text-slate-400 mt-1">
          Enter your capital and period — the AI will forecast exchange rates and optimise your deposit allocation.
        </p>
      </div>

      {/* ── Input Panel ── */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">Investment Parameters</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">

          {/* Budget */}
          <div className="space-y-1">
            <label className="text-xs text-slate-400">Total Capital (USD)</label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm">$</span>
              <input
                type="number"
                value={budget}
                onChange={e => setBudget(e.target.value)}
                placeholder="10000000"
                className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-7 pr-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Years */}
          <div className="space-y-1">
            <label className="text-xs text-slate-400">Investment Period (Years)</label>
            <input
              type="number"
              value={years}
              onChange={e => setYears(e.target.value)}
              min="0.25"
              max="30"
              step="0.5"
              placeholder="2"
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Currency Pair Filter */}
          <div className="space-y-1">
            <label className="text-xs text-slate-400">Currency Pair</label>
            <select
              value={pairFilter}
              onChange={e => setPair(e.target.value)}
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {PAIRS.map(p => (
                <option key={p} value={p}>{PAIR_LABELS[p]}</option>
              ))}
            </select>
          </div>

          {/* Risk Tolerance */}
          <div className="space-y-1">
            <label className="text-xs text-slate-400">Risk Tolerance</label>
            <select
              value={risk}
              onChange={e => setRisk(e.target.value as 'low' | 'medium' | 'high')}
              className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="low">Low — Conservative</option>
              <option value="medium">Medium — Balanced</option>
              <option value="high">High — Aggressive</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleCalculate}
          disabled={loading}
          className="mt-5 w-full md:w-auto bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-semibold text-sm rounded-lg px-8 py-2.5 transition-colors"
        >
          {loading ? 'Calculating…' : 'Calculate Investment Return'}
        </button>

        {error && <p className="mt-3 text-red-400 text-sm">{error}</p>}
      </div>

      {/* ── Results ── */}
      {result && (
        <>
          {/* Summary banner */}
          <div className="bg-gradient-to-r from-blue-900 to-slate-800 rounded-xl p-6 border border-blue-700">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <p className="text-xs text-blue-300 uppercase tracking-wide">Initial Capital</p>
                <p className="text-xl font-bold">${fmt(result.total_initial)}</p>
              </div>
              <div>
                <p className="text-xs text-blue-300 uppercase tracking-wide">Interest Earned</p>
                <p className="text-xl font-bold text-green-400">+${fmt(result.total_interest)}</p>
              </div>
              <div>
                <p className="text-xs text-blue-300 uppercase tracking-wide">FX Gain / Loss</p>
                <p className={`text-xl font-bold ${result.total_fx_gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.total_fx_gain >= 0 ? '+' : ''}${fmt(result.total_fx_gain)}
                </p>
              </div>
              <div>
                <p className="text-xs text-blue-300 uppercase tracking-wide">Total Gain</p>
                <p className={`text-xl font-bold ${result.total_gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.total_gain >= 0 ? '+' : ''}${fmt(result.total_gain)}
                </p>
              </div>
              <div>
                <p className="text-xs text-blue-300 uppercase tracking-wide">Projected Value</p>
                <p className="text-xl font-bold">${fmt(result.total_projected)}</p>
                <span className={`text-sm font-bold ${result.overall_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.overall_return_pct >= 0 ? '+' : ''}{result.overall_return_pct.toFixed(2)}% over {result.years}yr
                </span>
              </div>
            </div>

            <div className="mt-3 flex gap-3 flex-wrap">
              <span className={`text-xs px-3 py-1 rounded-full font-medium ${RISK_COLORS[result.risk_tolerance]}`}>
                {result.risk_tolerance.toUpperCase()} RISK
              </span>
              <span className="text-xs px-3 py-1 rounded-full font-medium bg-slate-700 text-slate-300">
                {result.pair_filter === 'ALL' ? 'All Currencies' : result.pair_filter.replace('_', '/')}
              </span>
              <span className="text-xs px-3 py-1 rounded-full font-medium bg-slate-700 text-slate-300">
                {result.years} year{result.years !== 1 ? 's' : ''} · {Math.round(result.years * 365)} days
              </span>
            </div>
          </div>

          {/* Per-currency cards */}
          <div>
            <h2 className="text-base font-semibold text-slate-300 mb-4">Breakdown by Currency</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.projections.map(p => <ProjectionCard key={p.pair} p={p} />)}
            </div>
          </div>

          {/* Exchange rate forecast table */}
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">
              LSTM Exchange Rate Forecast
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-slate-400 border-b border-slate-700">
                    <th className="text-left pb-3">Currency Pair</th>
                    <th className="text-right pb-3">Forecast Move</th>
                    <th className="text-right pb-3">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.forecast_returns).map(([pair, ret]) => (
                    <tr key={pair} className="border-b border-slate-700 last:border-0">
                      <td className="py-3 font-mono font-medium">{pair.replace('_', '/')}</td>
                      <td className="py-3 text-right">
                        <PctBadge value={ret * 100} />
                      </td>
                      <td className="py-3 text-right">
                        <span className={`text-xs px-2 py-1 rounded-full font-medium
                          ${ret >= 0 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                          {ret >= 0 ? 'APPRECIATE' : 'DEPRECIATE'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
