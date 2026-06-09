"""
Agent Orchestrator.
Coordinates all agents in a single async pipeline:

  MarketMonitor  →  alert detected
        ↓
  ForecastAgent  →  LSTM inference
        ↓
  DecisionAgent  →  LP optimization
        ↓
  RebalanceAgent →  simulate/execute trades

The orchestrator also supports a scheduled mode that runs the full
pipeline on a fixed interval (e.g. daily).
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.base_agent      import AgentStatus
from ml.agents.market_monitor  import MarketMonitorAgent
from ml.agents.forecast_agent  import ForecastAgent
from ml.agents.decision_agent  import DecisionAgent
from ml.agents.rebalance_agent import RebalanceAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parents[2] / "data" / "agent_logs"


class AgentOrchestrator:
    """
    Manages the lifecycle of all agents and runs the investment pipeline.

    Args:
        budget:           Total capital (USD)
        risk_tolerance:   "low" | "medium" | "high"
        alert_threshold:  Log-return shift that triggers the pipeline (default 0.5%)
        simulation:       If True, rebalancing is simulated (no real trades)
    """

    def __init__(
        self,
        budget:          float = 10_000.0,
        risk_tolerance:  str   = "medium",
        alert_threshold: float = 0.005,
        simulation:      bool  = True,
    ):
        self.monitor   = MarketMonitorAgent(threshold=alert_threshold, poll_interval=60)
        self.forecaster = ForecastAgent()
        self.decision   = DecisionAgent(
            budget=budget,
            risk_tolerance=risk_tolerance,
        )
        self.rebalancer = RebalanceAgent(simulation=simulation)

        self._pipeline_log: list[dict] = []

    # ── Single pipeline run ────────────────────────────────────────────────────

    async def run_pipeline(self, trigger: str = "manual") -> dict:
        """
        Execute one full Forecast → Decision → Rebalance cycle.

        Args:
            trigger: What triggered this run ("manual", "alert", "scheduled")

        Returns:
            Summary dict of the pipeline run
        """
        logger.info(f"Pipeline triggered by: {trigger}")
        start = datetime.utcnow()

        # Step 1 — Forecast
        forecasts = await self.forecaster.run()
        if not forecasts:
            logger.warning("No forecasts available. Ensure models are trained.")
            return {"status": "no_forecasts", "trigger": trigger}

        # Step 2 — Decision
        decision = await self.decision.run(forecasts)

        # Step 3 — Rebalance
        rebalance_result = await self.rebalancer.run(decision)

        # Propagate executed allocation back to DecisionAgent so future drift
        # checks compare against what was actually deployed, not an empty baseline.
        if rebalance_result.get("rebalanced"):
            self.decision.current_portfolio = decision["allocations"]

        summary = {
            "trigger":          trigger,
            "timestamp":        start.isoformat(),
            "forecasts":        {p: f"{v['forecast_return']*100:.3f}%" for p, v in forecasts.items()},
            "allocations":      {p: f"{w*100:.1f}%" for p, w in decision["allocations"].items()},
            "expected_return":  f"{decision['expected_return_pct']*100:.2f}%",
            "rebalanced":       rebalance_result.get("rebalanced", False),
            "trades":           rebalance_result.get("trades", {}),
        }

        self._pipeline_log.append(summary)
        self._save_log(summary)

        logger.info("Pipeline complete.")
        logger.info(f"  Expected return : {summary['expected_return']}")
        logger.info(f"  Rebalanced      : {summary['rebalanced']}")
        return summary

    # ── Scheduled mode ─────────────────────────────────────────────────────────

    async def run_scheduled(self, interval_seconds: int = 86400, max_runs: int | None = None) -> None:
        """
        Run the pipeline on a schedule (default: daily).

        Args:
            interval_seconds: Seconds between pipeline runs (86400 = 24h)
            max_runs:         Stop after this many runs (None = run forever)
        """
        logger.info(f"Scheduled mode: interval={interval_seconds}s  max_runs={max_runs}")
        runs = 0

        while True:
            await self.run_pipeline(trigger="scheduled")
            runs += 1
            if max_runs is not None and runs >= max_runs:
                break
            logger.info(f"Next run in {interval_seconds}s …")
            await asyncio.sleep(interval_seconds)

    # ── Alert-driven mode ──────────────────────────────────────────────────────

    async def run_with_monitor(self, max_polls: int = 5) -> None:
        """
        Run the market monitor alongside the pipeline.
        Triggers a pipeline run as a concurrent task whenever an alert is detected.

        Args:
            max_polls: Number of monitor polls before stopping (for testing)
        """
        self.monitor.status = AgentStatus.RUNNING
        for _ in range(max_polls):
            prices = await self.monitor.fetch_latest()
            alerts = self.monitor.detect_shifts(prices)
            self.monitor.last_prices.update(prices)

            if alerts:
                logger.info("Alert detected! Triggering pipeline...")
                asyncio.create_task(self.run_pipeline(trigger="alert"))

            await asyncio.sleep(self.monitor.poll_interval)

        self.monitor.status = AgentStatus.IDLE

    # ── Status ─────────────────────────────────────────────────────────────────

    def status_report(self) -> dict:
        return {
            "MarketMonitor":  self.monitor.status,
            "ForecastAgent":  self.forecaster.status,
            "DecisionAgent":  self.decision.status,
            "RebalanceAgent": self.rebalancer.status,
            "pipeline_runs":  len(self._pipeline_log),
        }

    def get_full_log(self) -> list[dict]:
        logs = []
        for agent in [self.monitor, self.forecaster, self.decision, self.rebalancer]:
            logs.extend(agent.log)
        logs.sort(key=lambda x: x["timestamp"])
        return logs

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_log(self, summary: dict) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"pipeline_{ts}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    orchestrator = AgentOrchestrator(budget=10_000, risk_tolerance="medium", simulation=True)
    result = asyncio.run(orchestrator.run_pipeline(trigger="manual"))
    print("\nPipeline summary:")
    print(json.dumps(result, indent=2))
