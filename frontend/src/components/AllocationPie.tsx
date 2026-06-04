import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORS = ['#0ea5e9', '#f59e0b', '#10b981']

interface Props {
  allocations: Record<string, number>   // pair → weight_pct
  title?: string
}

export default function AllocationPie({ allocations, title }: Props) {
  const data = Object.entries(allocations).map(([pair, value]) => ({
    name: pair.replace('_', '/'),
    value: parseFloat(value.toFixed(2)),
  }))

  return (
    <div className="bg-surface rounded-xl p-4">
      {title && <h3 className="text-sm font-semibold text-muted mb-3">{title}</h3>}
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name} ${value}%`} labelLine={false}>
            {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Pie>
          <Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
