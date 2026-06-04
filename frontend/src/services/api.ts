import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// Attach JWT token to every request if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── Auth ──────────────────────────────────────────────────────────────────────
export const login = (username: string, password: string) =>
  api.post<{ access_token: string }>('/auth/login', { username, password })

// ── Forecast ──────────────────────────────────────────────────────────────────
export const getAllForecasts = () =>
  api.get<AllForecastsResponse>('/forecast/')

export const getPairForecast = (pair: string) =>
  api.get<ForecastItem>(`/forecast/${pair}`)

export const getPairHistory = (pair: string) =>
  api.get<{ pair: string; actual: number[]; predicted: number[] }>(`/forecast/${pair}/history`)

// ── Portfolio ─────────────────────────────────────────────────────────────────
export const optimizePortfolio = (body: OptimizeRequest) =>
  api.post<PortfolioResponse>('/portfolio/optimize', body)

export const getBenchmark = (params?: BenchmarkParams) =>
  api.get<BenchmarkRow[]>('/portfolio/benchmark', { params })

// ── Agents ────────────────────────────────────────────────────────────────────
export const getAgentStatus = () =>
  api.get<AgentStatus>('/agents/status')

export const getAgentLog = (limit = 50) =>
  api.get<AgentLogEntry[]>(`/agents/log?limit=${limit}`)

export const runPipeline = (body: PipelineRunRequest) =>
  api.post<PipelineSummary>('/agents/run', body)

// ── Investment Calculator ─────────────────────────────────────────────────────
export const calculateInvestment = (body: InvestmentRequest) =>
  api.post<InvestmentResponse>('/investment/calculate', body)

// ── Types ─────────────────────────────────────────────────────────────────────
export interface ForecastItem {
  pair: string
  predicted_next: number
  last_actual: number
  forecast_return: number
  forecast_return_pct: string
}

export interface AllForecastsResponse {
  forecasts: Record<string, ForecastItem>
  generated_at: string
}

export interface OptimizeRequest {
  budget?: number
  risk_tolerance?: 'low' | 'medium' | 'high'
  min_weight?: number
  max_weight?: number
  investment_days?: number
}

export interface AllocationDetail {
  weight_pct: number
  amount_usd: number
}

export interface PortfolioResponse {
  allocations: Record<string, AllocationDetail>
  expected_return_pct: number
  expected_amount: number
  budget: number
  risk_tolerance: string
  solver_status: string
  generated_at: string
}

export interface BenchmarkParams {
  budget?: number
  investment_days?: number
  risk_tolerance?: string
}

export interface BenchmarkRow {
  strategy: string
  expected_return_pct: number
  expected_amount: number
  allocations: Record<string, number>
}

export interface AgentStatus {
  MarketMonitor: string
  ForecastAgent: string
  DecisionAgent: string
  RebalanceAgent: string
  pipeline_runs: number
}

export interface AgentLogEntry {
  timestamp: string
  agent: string
  action: string
  detail: Record<string, unknown>
}

export interface PipelineRunRequest {
  trigger?: string
  budget?: number
  risk_tolerance?: string
  simulation?: boolean
}

export interface PipelineSummary {
  trigger: string
  timestamp: string
  forecasts: Record<string, string>
  allocations: Record<string, string>
  expected_return: string
  rebalanced: boolean
}

export interface InvestmentRequest {
  budget:         number
  years:          number
  risk_tolerance: 'low' | 'medium' | 'high'
  pair_filter:    string
}

export interface PairProjection {
  pair:             string
  label:            string
  deposit_usd:      number
  weight_pct:       number
  annual_rate_pct:  number
  interest_usd:     number
  fx_return_pct:    number
  fx_gain_loss_usd: number
  total_gain_usd:   number
  projected_usd:    number
  total_return_pct: number
  direction:        'up' | 'down'
}

export interface InvestmentResponse {
  budget:             number
  years:              number
  risk_tolerance:     string
  pair_filter:        string
  projections:        PairProjection[]
  total_initial:      number
  total_interest:     number
  total_fx_gain:      number
  total_gain:         number
  total_projected:    number
  overall_return_pct: number
  forecast_returns:   Record<string, number>
}

export default api
