"""
Pydantic request/response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field


# ── Auth ───────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── Forecast ───────────────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    pair:             str
    predicted_next:   float
    last_actual:      float
    forecast_return:  float
    forecast_return_pct: str

class AllForecastsResponse(BaseModel):
    forecasts:  dict[str, ForecastResponse]
    generated_at: str


# ── Portfolio ──────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    budget:          float = Field(default=10_000.0, gt=0, description="Total capital in USD")
    risk_tolerance:  str   = Field(default="medium",  pattern="^(low|medium|high)$")
    min_weight:      float = Field(default=0.05, ge=0.0, le=1.0)
    max_weight:      float = Field(default=0.60, ge=0.0, le=1.0)
    investment_days: int   = Field(default=365, gt=0)

class AllocationDetail(BaseModel):
    weight_pct: float
    amount_usd: float

class PortfolioResponse(BaseModel):
    allocations:          dict[str, AllocationDetail]
    expected_return_pct:  float
    expected_amount:      float
    budget:               float
    risk_tolerance:       str
    solver_status:        str
    generated_at:         str

class BenchmarkRow(BaseModel):
    strategy:             str
    expected_return_pct:  float
    expected_amount:      float
    allocations:          dict[str, float]


# ── Agents ─────────────────────────────────────────────────────────────────────

class AgentStatusResponse(BaseModel):
    MarketMonitor:  str
    ForecastAgent:  str
    DecisionAgent:  str
    RebalanceAgent: str
    pipeline_runs:  int

class AgentLogEntry(BaseModel):
    timestamp: str
    agent:     str
    action:    str
    detail:    dict

class PipelineRunRequest(BaseModel):
    trigger:         str   = "manual"
    budget:          float = 10_000.0
    risk_tolerance:  str   = "medium"
    simulation:      bool  = True
