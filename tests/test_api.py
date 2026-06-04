"""
API integration tests using FastAPI's TestClient.
Run with: python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_root_returns_ok(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ── Auth ───────────────────────────────────────────────────────────────────────

class TestAuth:

    def test_login_success(self):
        r = client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
        assert r.status_code == 200
        assert "access_token" in r.json()
        assert r.json()["token_type"] == "bearer"

    def test_login_wrong_password(self):
        r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401

    def test_login_unknown_user(self):
        r = client.post("/api/auth/login", json={"username": "nobody", "password": "x"})
        assert r.status_code == 401


# ── Forecast ───────────────────────────────────────────────────────────────────

class TestForecast:

    def test_invalid_pair_returns_400(self):
        r = client.get("/api/forecast/INVALID")
        assert r.status_code == 400

    def test_valid_pair_format_accepted(self):
        # Will return 503 if model not trained — that's expected in CI
        r = client.get("/api/forecast/EUR_USD")
        assert r.status_code in (200, 503)

    def test_all_forecasts_endpoint_exists(self):
        r = client.get("/api/forecast/")
        assert r.status_code in (200, 503)


# ── Portfolio ──────────────────────────────────────────────────────────────────

class TestPortfolio:

    def test_optimize_with_valid_body(self):
        body = {
            "budget": 10000,
            "risk_tolerance": "medium",
            "min_weight": 0.05,
            "max_weight": 0.60,
            "investment_days": 365,
        }
        r = client.post("/api/portfolio/optimize", json=body)
        # 200 if models trained, 503 if not
        assert r.status_code in (200, 503)

    def test_optimize_invalid_risk_tolerance(self):
        body = {"budget": 5000, "risk_tolerance": "extreme"}
        r = client.post("/api/portfolio/optimize", json=body)
        assert r.status_code == 422   # Pydantic validation error

    def test_optimize_negative_budget(self):
        body = {"budget": -100, "risk_tolerance": "medium"}
        r = client.post("/api/portfolio/optimize", json=body)
        assert r.status_code == 422

    def test_benchmark_endpoint_exists(self):
        r = client.get("/api/portfolio/benchmark")
        assert r.status_code in (200, 503)


# ── Agents ─────────────────────────────────────────────────────────────────────

class TestAgents:

    def test_status_endpoint(self):
        r = client.get("/api/agents/status")
        assert r.status_code == 200
        data = r.json()
        assert "MarketMonitor" in data
        assert "ForecastAgent" in data
        assert "pipeline_runs" in data

    def test_log_endpoint(self):
        r = client.get("/api/agents/log")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_log_limit_parameter(self):
        r = client.get("/api/agents/log?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5
