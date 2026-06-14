# Multicurrency Investment AI

Optimization of investment return for multicurrency accounts using Agentic AI — targeting Sri Lankan bank depositors. Combines LSTM forecasting, linear programming portfolio optimization, and a multi-agent pipeline built with LangChain.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Environment Variables](#environment-variables)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Running the ML Pipeline](#running-the-ml-pipeline)
- [Project Structure](#project-structure)

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| npm | 9+ |
| Git | any |

A PostgreSQL database is optional — the backend defaults to SQLite if `DATABASE_URL` is not set.

---

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/555Salman/investment-return.git
cd investment-return
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Environment Variables

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

Open `.env` and update the following:

```env
# Required in production — generate with:
# python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your-256-bit-random-secret-here

# Demo login credentials
DEMO_USERNAME=admin
DEMO_PASSWORD=your-password-here

# OpenAI key for LangChain agents
OPENAI_API_KEY=sk-...

# Optional — leave unset to use the default SQLite database
# DATABASE_URL=postgresql://user:password@localhost:5432/multicurrency_db

# Optional — only needed if Redis is used
# REDIS_URL=redis://localhost:6379

# Optional — only needed for production deployments
# VITE_API_BASE_URL=https://your-backend.com/api
```

> **Note:** `SECRET_KEY` defaults to an insecure placeholder in development (`debug=True`). It **must** be set to a random value before any production deployment.

---

## Running the Backend

The backend is a FastAPI application served by Uvicorn.

```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Health check (`{"status": "ok"}`) |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/redoc` | ReDoc API reference |

### Available API routes

| Prefix | Description |
|--------|-------------|
| `/api/auth` | Login — returns a JWT bearer token |
| `/api/forecast` | LSTM next-step forecasts per currency pair |
| `/api/portfolio` | LP portfolio optimization |
| `/api/investment` | Full investment projection calculator |
| `/api/agents` | Agent status, log, and pipeline trigger |

### Obtaining a token (first login)

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password-here"}'
```

---

## Running the Frontend

The frontend is a React + TypeScript app powered by Vite. It proxies all `/api` and `/ws` requests to the backend at `localhost:8000`, so the backend must be running first.

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

### Other frontend commands

```bash
# Type-check and build for production
npm run build

# Preview the production build locally
npm run preview
```

---

## Running the ML Pipeline

Use `run_pipeline.py` to execute all five project phases from the command line:

```bash
# Run all phases (preprocess → train → backtest → optimise → agents)
python run_pipeline.py

# Run only preprocessing and training
python run_pipeline.py --phases 1 2

# Run with custom training settings, skip the async agent phase
python run_pipeline.py --epochs 100 --patience 15 --skip-agents
```

| Phase | Description |
|-------|-------------|
| 1 | Data preprocessing — clean, feature-engineer, and create LSTM sequences |
| 2 | LSTM training — one model per currency pair with early stopping |
| 3 | Walk-forward backtest — LP vs equal-weight vs buy-and-hold |
| 4 | Portfolio optimisation — LP snapshot on latest forecasts |
| 5 | Agentic AI dry run — full multi-agent pipeline in simulation mode |

### Training a single pair manually

```bash
python ml/training/train.py --pair_name EUR_USD --epochs 50 --seed 42
```

### Running the backtest only

```bash
python ml/training/backtest.py --budget 10000 --risk_tolerance medium
```

---

## Project Structure

```
.
├── backend/               FastAPI backend
│   ├── core/              Config, security, database
│   ├── models/            Pydantic schemas
│   ├── routers/           API route handlers
│   └── services/          Business logic layer
├── frontend/              React + TypeScript + Vite
│   └── src/
│       ├── components/    Reusable UI components
│       ├── hooks/         Custom React hooks (WebSocket, etc.)
│       ├── pages/         Page-level components
│       └── services/      Axios API client
├── ml/                    ML pipeline
│   ├── agents/            Multi-agent system (orchestrator + 4 agents)
│   ├── data_pipeline/     Ingestion, preprocessing, PyTorch datasets
│   ├── models/            LSTM architecture, ARIMA baseline, utils
│   ├── optimization/      LP portfolio optimizer, risk manager
│   └── training/          Train loop, backtest engine, evaluate
├── notebooks/             Jupyter notebooks (exploration → prototype)
├── data/                  Raw CSVs, processed arrays, model checkpoints
├── tests/                 pytest test suite
├── requirements.txt       Python dependencies
├── run_pipeline.py        End-to-end pipeline runner
└── .env.example           Environment variable template
```
