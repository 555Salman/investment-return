# Project Plan: Optimization of Investment Return of Multicurrency Accounts using Agentic AI in Sri Lanka

---

## Overview

This project combines machine learning (LSTM-based forecasting), linear programming (portfolio optimization), and agentic AI to build an intelligent multicurrency investment platform targeting Sri Lankan bank depositors.

---

## Project Structure

```
multicurrency-investment-ai/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Downloaded from Yahoo Finance / Bank of Ceylon
│   └── processed/                 # Cleaned, normalized time-series data
│
├── notebooks/                     # Exploratory and experimental Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_volatility_analysis.ipynb
│   ├── 03_lstm_forecasting.ipynb
│   ├── 04_optimization_model.ipynb
│   └── 05_agentic_ai_prototype.ipynb
│
├── ml/                            # Core ML pipeline
│   ├── data_pipeline/
│   │   ├── ingestion.py           # Yahoo Finance / CSV data loader
│   │   ├── preprocessing.py       # Forward fill, normalization, feature engineering
│   │   └── dataset.py             # PyTorch/TF Dataset wrappers
│   ├── models/
│   │   ├── lstm_model.py          # LSTM architecture definition
│   │   ├── arima_baseline.py      # ARIMA comparison model
│   │   └── model_utils.py         # Save/load model helpers
│   ├── training/
│   │   ├── train.py               # Training loop
│   │   ├── hyperparameter_tuning.py  # Bayesian Opt + Grid Search
│   │   └── evaluate.py            # RMSE, MAE evaluation
│   ├── optimization/
│   │   ├── portfolio_optimizer.py # Linear programming capital allocation
│   │   └── risk_manager.py        # Volatility constraints handler
│   └── agents/
│       ├── market_monitor.py      # Real-time market monitoring agent
│       ├── decision_agent.py      # Investment decision-making agent
│       ├── rebalance_agent.py     # Portfolio rebalancing agent
│       └── agent_orchestrator.py  # Multi-agent coordinator
│
├── backend/                       # FastAPI backend
│   ├── main.py                    # App entry point
│   ├── routers/
│   │   ├── forecast.py            # /api/forecast endpoints
│   │   ├── portfolio.py           # /api/portfolio endpoints
│   │   ├── agents.py              # /api/agents endpoints
│   │   └── auth.py                # /api/auth endpoints
│   ├── services/
│   │   ├── forecast_service.py    # Wraps ML forecasting pipeline
│   │   ├── portfolio_service.py   # Wraps optimization engine
│   │   └── agent_service.py       # Manages agent lifecycle
│   ├── models/                    # Pydantic schemas
│   │   ├── forecast_schema.py
│   │   └── portfolio_schema.py
│   └── core/
│       ├── config.py              # Environment config
│       └── database.py            # DB connection (PostgreSQL)
│
├── frontend/                      # React + TypeScript frontend
│   ├── public/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx      # Main investor dashboard
│   │   │   ├── Forecast.tsx       # Exchange rate forecast charts
│   │   │   ├── Portfolio.tsx      # Portfolio allocation view
│   │   │   ├── Agents.tsx         # Agent activity monitor
│   │   │   └── Login.tsx
│   │   ├── components/
│   │   │   ├── CurrencyChart.tsx  # Recharts/Chart.js line charts
│   │   │   ├── AllocationPie.tsx  # Portfolio pie chart
│   │   │   ├── RiskMeter.tsx      # Volatility risk indicator
│   │   │   └── AgentLog.tsx       # Live agent decision log
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts    # Real-time data via WebSocket
│   │   ├── services/
│   │   │   └── api.ts             # Axios API client
│   │   └── store/
│   │       └── portfolioStore.ts  # Zustand/Redux state
│   └── package.json
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_lstm.py
│   ├── test_optimizer.py
│   └── test_api.py
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Phase 1: Data Pipeline

**Goal:** Collect, clean, and prepare historical exchange rate data.

### Tasks
- [ ] Download 5 years (2020–2025) of daily OHLC data for USD/EUR, USD/AUD, USD/NZD from Yahoo Finance using `yfinance`
- [ ] Supplement with Bank of Ceylon published rates where available
- [ ] Handle missing values using forward fill
- [ ] Compute log returns: `log(Close_t / Close_{t-1})`
- [ ] Normalize features using MinMaxScaler / StandardScaler
- [ ] Split into train (80%), validation (10%), test (10%) sets
- [ ] Create sliding window sequences for LSTM input (e.g., 60-day lookback)

### Tech Stack
- `yfinance`, `pandas`, `numpy`, `scikit-learn`

---

## Phase 2: Volatility Analysis

**Goal:** Understand risk associated with each currency pair.

### Tasks
- [ ] Compute rolling standard deviation of log returns (30-day, 60-day windows)
- [ ] Plot volatility trends over the 5-year period
- [ ] Identify high-volatility periods (e.g., COVID-19, Sri Lanka economic crisis 2022)
- [ ] Compute correlation matrix across the three currency pairs
- [ ] Document findings to inform optimization constraints

### Outputs
- Volatility charts per currency pair
- Correlation heatmap
- Risk profile report

---

## Phase 3: LSTM Forecasting Model

**Goal:** Build a reliable exchange rate forecasting model.

### Model Architecture
```
Input (sequence_length=60, features=5)  →  LSTM(128, return_sequences=True)
→  Dropout(0.2)  →  LSTM(64)  →  Dropout(0.2)  →  Dense(32)  →  Dense(1)
```

### Tasks
- [ ] Implement LSTM model in PyTorch (or TensorFlow/Keras)
- [ ] Train separate models per currency pair (USD/EUR, USD/AUD, USD/NZD)
- [ ] Implement Bayesian Optimization (using `optuna`) for hyperparameter tuning:
  - Learning rate, hidden units, dropout rate, sequence length, batch size
- [ ] Implement Grid Search as secondary comparison
- [ ] Implement ARIMA baseline using `statsmodels`
- [ ] Evaluate models: RMSE, MAE
- [ ] Compare LSTM vs ARIMA performance
- [ ] Save best model weights per currency pair

### Tech Stack
- `PyTorch` or `TensorFlow`, `optuna`, `statsmodels`, `matplotlib`

---

## Phase 4: Portfolio Optimization

**Goal:** Allocate capital across currencies to maximize risk-adjusted returns.

### Linear Programming Model

**Objective:**
```
Maximize:  Σ (forecast_return_i + interest_rate_i) × allocation_i
```

**Subject to:**
```
Σ allocation_i = 1                     (budget constraint)
allocation_i >= 0                      (no short selling)
volatility_i × allocation_i <= limit_i (risk constraint per currency)
allocation_i <= max_weight             (diversification cap, e.g., 60%)
```

### Tasks
- [ ] Implement LP model using `scipy.optimize.linprog` or `PuLP`
- [ ] Integrate LSTM forecasted returns as input
- [ ] Apply rolling volatility as risk constraint
- [ ] Incorporate Sri Lankan bank deposit interest rates per currency
- [ ] Test with multiple budget scenarios (e.g., $1,000 / $10,000)
- [ ] Benchmark against equal-weight and buy-and-hold strategies

### Tech Stack
- `scipy`, `PuLP`, `numpy`

---

## Phase 5: Agentic AI Layer

**Goal:** Enable autonomous, real-time portfolio management.

### Agent Design

| Agent | Role |
|---|---|
| Market Monitor Agent | Polls live exchange rate data, detects significant shifts |
| Forecast Agent | Triggers LSTM re-inference on new data |
| Decision Agent | Runs LP optimizer, proposes allocation changes |
| Rebalance Agent | Executes or recommends portfolio rebalancing |
| Orchestrator | Coordinates agents, manages state, logs decisions |

### Tasks
- [ ] Design agent communication protocol (event-driven / message queue)
- [ ] Implement agents using `LangChain Agents` or custom Python async agents
- [ ] Enable agents to call LSTM forecasting and LP optimization as tools
- [ ] Implement decision thresholds (e.g., rebalance if drift > 5%)
- [ ] Add continuous learning: retrain LSTM periodically on new data
- [ ] Log all agent decisions with timestamps and rationale

### Tech Stack
- `LangChain`, `asyncio`, `Redis` (message queue), `APScheduler`

---

## Phase 6: Backend API

**Goal:** Serve ML outputs and agent state via a REST + WebSocket API.

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/forecast/{pair}` | Get 7/14/30-day price forecast |
| GET | `/api/portfolio/optimize` | Get LP-optimized allocation |
| GET | `/api/portfolio/current` | Get current portfolio state |
| POST | `/api/portfolio/update` | Update investment parameters |
| GET | `/api/agents/status` | Get all agent statuses |
| GET | `/api/agents/log` | Get agent decision history |
| WS | `/ws/market` | Real-time exchange rate stream |
| WS | `/ws/agents` | Real-time agent activity stream |

### Tasks
- [ ] Set up FastAPI application with CORS, auth (JWT)
- [ ] Integrate ML services (forecasting, optimization, agents) as injectable services
- [ ] Set up PostgreSQL for storing portfolio history and agent logs
- [ ] Implement WebSocket endpoints for real-time dashboard updates
- [ ] Write unit tests for all endpoints
- [ ] Containerize with Docker

### Tech Stack
- `FastAPI`, `PostgreSQL`, `SQLAlchemy`, `JWT`, `Docker`

---

## Phase 7: Frontend Dashboard

**Goal:** Provide an investor-facing interface to visualize forecasts, allocations, and agent activity.

### Pages and Features

#### Dashboard (Home)
- Portfolio summary cards (total value, current return %, risk level)
- Active currency allocations
- Live exchange rate ticker

#### Forecast Page
- Interactive line charts showing historical + predicted rates per currency
- Forecast horizon selector (7 / 14 / 30 days)
- Confidence interval bands

#### Portfolio Page
- Pie chart of current capital allocation
- Optimization inputs (budget, risk tolerance, investment duration)
- Run Optimizer button → triggers LP engine
- Benchmark comparison table (AI vs equal-weight vs hold)

#### Agents Page
- Agent status cards (running / idle / alert)
- Real-time agent decision log feed
- Agent override controls (pause / resume)

#### Settings / Profile
- Investment preferences
- Notification thresholds

### Tasks
- [ ] Set up React + TypeScript + Tailwind CSS project (Vite)
- [ ] Build reusable chart components using `Recharts`
- [ ] Implement real-time updates via WebSocket hook
- [ ] Build API client using `axios`
- [ ] Implement authentication flow (Login / JWT storage)
- [ ] Ensure mobile-responsive layout
- [ ] Deploy frontend (Vercel / Netlify)

### Tech Stack
- `React`, `TypeScript`, `Tailwind CSS`, `Recharts`, `Axios`, `Zustand`

---

## Phase 8: Evaluation

**Goal:** Validate both forecasting and investment performance.

### Forecasting Evaluation
| Metric | Target |
|---|---|
| RMSE | Minimize (compare vs ARIMA baseline) |
| MAE | Minimize |
| Directional Accuracy | > 55% |

### Investment Performance Evaluation
| Metric | Baseline |
|---|---|
| Total Return | Compare vs equal-weight |
| Sharpe Ratio | > 1.0 preferred |
| Max Drawdown | Lower than buy-and-hold |
| Portfolio Stability | Rolling volatility comparison |

### Tasks
- [ ] Run backtesting over 2024 data (out-of-sample)
- [ ] Generate performance report (charts + tables)
- [ ] Document findings for research paper

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Data Collection | `yfinance`, `pandas` |
| ML Framework | `PyTorch` or `TensorFlow` |
| Hyperparameter Tuning | `Optuna` (Bayesian), Grid Search |
| Optimization | `PuLP` / `scipy` |
| Agentic AI | `LangChain` / custom agents |
| Backend | `FastAPI`, `PostgreSQL`, `Redis` |
| Frontend | `React`, `TypeScript`, `Tailwind CSS` |
| Charts | `Recharts` |
| Containerization | `Docker`, `Docker Compose` |
| CI/CD | `GitHub Actions` |
| Deployment | Backend: `Railway` / `Render`, Frontend: `Vercel` |

---

## Development Timeline

| Phase | Description | Duration |
|---|---|---|
| 1 | Literature review + Data collection + Preprocessing | Week 1–2 |
| 2 | Volatility analysis + EDA | Week 3 |
| 3 | LSTM model development + tuning + baseline comparison | Week 4–6 |
| 4 | Portfolio optimization model | Week 7–8 |
| 5 | Agentic AI design + implementation | Week 9–11 |
| 6 | Backend API development | Week 12–13 |
| 7 | Frontend dashboard | Week 14–15 |
| 8 | Integration testing + backtesting + evaluation | Week 16–17 |
| 9 | Documentation + research paper writing + presentation | Week 18–20 |

---

## Evaluation Checklist

- [ ] LSTM outperforms ARIMA on RMSE and MAE
- [ ] LP optimizer outperforms equal-weight strategy on risk-adjusted return
- [ ] Agents successfully rebalance portfolio in simulated real-time scenarios
- [ ] Dashboard displays live forecasts and agent decisions
- [ ] Full backtesting report generated

---

## Research Deliverables

1. **Trained ML Models** — LSTM models per currency pair with saved weights
2. **Optimization Engine** — LP model integrated with forecasts
3. **Agentic AI System** — Multi-agent framework for autonomous decisions
4. **Web Platform** — Full-stack investor dashboard
5. **Research Paper** — Documenting methodology, results, and contributions
6. **Presentation** — Final demo and slides
