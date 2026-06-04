import { NavLink } from 'react-router-dom'
import { usePortfolioStore } from '../store/portfolioStore'

const links = [
  { to: '/',           label: 'Dashboard'   },
  { to: '/forecast',   label: 'Forecast'    },
  { to: '/portfolio',  label: 'Portfolio'   },
  { to: '/investment', label: 'Calculator'  },
  { to: '/agents',     label: 'Agents'      },
]

export default function Navbar() {
  const { token, setToken } = usePortfolioStore()

  return (
    <nav className="bg-surface border-b border-slate-700 px-6 py-3 flex items-center justify-between">
      <span className="font-bold text-accent tracking-tight">
        Multicurrency AI
      </span>
      <div className="flex gap-6">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `text-sm transition-colors ${isActive ? 'text-accent font-semibold' : 'text-muted hover:text-white'}`
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
      {token && (
        <button
          onClick={() => setToken(null)}
          className="text-xs text-muted hover:text-white transition-colors"
        >
          Logout
        </button>
      )}
    </nav>
  )
}
