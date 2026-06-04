"""
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.database import Base, engine
from backend.routers import auth, forecast, portfolio, agents
from backend.routers import investment

# Create DB tables on startup (no-op if they already exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agentic AI system for multicurrency investment optimization in Sri Lanka.",
)

# Allow the React frontend (localhost:5173) during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(forecast.router)
app.include_router(portfolio.router)
app.include_router(agents.router)
app.include_router(investment.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
