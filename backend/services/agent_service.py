"""
Agent service — manages the AgentOrchestrator singleton for FastAPI.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.agent_orchestrator import AgentOrchestrator


class AgentService:
    def __init__(self):
        self._orchestrator = AgentOrchestrator(simulation=True)
        # Serialise concurrent pipeline requests so budget/risk/simulation mutations
        # on the shared agents never race between simultaneous HTTP requests.
        self._lock = asyncio.Lock()

    async def run_pipeline(
        self,
        trigger: str,
        budget: float,
        risk_tolerance: str,
        simulation: bool = True,
    ) -> dict:
        async with self._lock:
            self._orchestrator.decision.budget         = budget
            self._orchestrator.decision.risk_tolerance = risk_tolerance
            self._orchestrator.rebalancer.simulation   = simulation
            return await self._orchestrator.run_pipeline(trigger=trigger)

    def get_status(self) -> dict:
        return self._orchestrator.status_report()

    def get_log(self, limit: int = 50) -> list[dict]:
        return self._orchestrator.get_full_log()[-limit:]


agent_service = AgentService()
