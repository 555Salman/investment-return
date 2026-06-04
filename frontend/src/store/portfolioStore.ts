import { create } from 'zustand'
import type { ForecastItem, PortfolioResponse, AgentStatus, AgentLogEntry } from '../services/api'

interface PortfolioStore {
  // Forecast
  forecasts: Record<string, ForecastItem>
  forecastsLoading: boolean
  setForecasts: (f: Record<string, ForecastItem>) => void
  setForecastsLoading: (v: boolean) => void

  // Portfolio
  portfolio: PortfolioResponse | null
  portfolioLoading: boolean
  setPortfolio: (p: PortfolioResponse) => void
  setPortfolioLoading: (v: boolean) => void

  // Agents
  agentStatus: AgentStatus | null
  agentLog: AgentLogEntry[]
  setAgentStatus: (s: AgentStatus) => void
  appendAgentLog: (entries: AgentLogEntry[]) => void

  // Auth
  token: string | null
  setToken: (t: string | null) => void
}

export const usePortfolioStore = create<PortfolioStore>((set) => ({
  forecasts: {},
  forecastsLoading: false,
  setForecasts: (forecasts) => set({ forecasts }),
  setForecastsLoading: (forecastsLoading) => set({ forecastsLoading }),

  portfolio: null,
  portfolioLoading: false,
  setPortfolio: (portfolio) => set({ portfolio }),
  setPortfolioLoading: (portfolioLoading) => set({ portfolioLoading }),

  agentStatus: null,
  agentLog: [],
  setAgentStatus: (agentStatus) => set({ agentStatus }),
  appendAgentLog: (entries) =>
    set((s) => ({ agentLog: [...s.agentLog, ...entries].slice(-200) })),

  token: localStorage.getItem('token'),
  setToken: (token) => {
    if (token) localStorage.setItem('token', token)
    else localStorage.removeItem('token')
    set({ token })
  },
}))
