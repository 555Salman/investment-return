import { Routes, Route, Navigate } from 'react-router-dom'
import { usePortfolioStore } from './store/portfolioStore'
import Navbar     from './components/Navbar'
import Login      from './pages/Login'
import Dashboard  from './pages/Dashboard'
import Forecast   from './pages/Forecast'
import Portfolio  from './pages/Portfolio'
import Agents     from './pages/Agents'
import Investment from './pages/Investment'

function ProtectedLayout({ children }: { children: React.ReactNode }) {
  const { token } = usePortfolioStore()
  if (!token) return <Navigate to="/login" replace />
  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <Navbar />
      <main className="max-w-6xl mx-auto">{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<ProtectedLayout><Dashboard /></ProtectedLayout>} />
      <Route path="/forecast"  element={<ProtectedLayout><Forecast /></ProtectedLayout>} />
      <Route path="/portfolio" element={<ProtectedLayout><Portfolio /></ProtectedLayout>} />
      <Route path="/agents"      element={<ProtectedLayout><Agents /></ProtectedLayout>} />
      <Route path="/investment"  element={<ProtectedLayout><Investment /></ProtectedLayout>} />
    </Routes>
  )
}
