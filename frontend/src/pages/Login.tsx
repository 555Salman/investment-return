import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { login } from '../services/api'
import { usePortfolioStore } from '../store/portfolioStore'

export default function Login() {
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const { setToken }            = usePortfolioStore()
  const navigate                = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await login(username, password)
      setToken(res.data.access_token)
      navigate('/')
    } catch {
      setError('Invalid credentials. Try admin / password123')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-900">
      <form onSubmit={handleSubmit} className="bg-surface rounded-2xl p-8 w-full max-w-sm space-y-4 shadow-xl">
        <h1 className="text-xl font-bold text-white">Sign In</h1>
        <p className="text-xs text-muted">Multicurrency Investment AI Dashboard</p>

        <input
          className="w-full bg-slate-800 rounded-lg px-4 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-accent"
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
        <input
          type="password"
          className="w-full bg-slate-800 rounded-lg px-4 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-accent"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />

        {error && <p className="text-red-400 text-xs">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-accent hover:bg-blue-400 text-white rounded-lg py-2 text-sm font-semibold transition-colors disabled:opacity-50"
        >
          {loading ? 'Signing in…' : 'Sign In'}
        </button>
      </form>
    </div>
  )
}
