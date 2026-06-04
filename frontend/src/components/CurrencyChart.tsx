import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

interface Props {
  data: { date: string; actual: number; predicted?: number }[]
  pair: string
}

export default function CurrencyChart({ data, pair }: Props) {
  return (
    <div className="bg-surface rounded-xl p-4">
      <h3 className="text-sm font-semibold text-muted mb-3">
        {pair.replace('_', '/')} — Price
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="actual"    stroke="#0ea5e9" dot={false} strokeWidth={1.5} name="Actual" />
          <Line type="monotone" dataKey="predicted" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="Predicted" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
