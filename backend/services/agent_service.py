"""
Agent service — manages the AgentOrchestrator singleton for FastAPI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.agent_orchestrator import AgentOrchestrator


class AgentService:
    def __init__(self):
        self._orchestrator = AgentOrchestrator(simulation=True)

    async def run_pipeline(self, trigger: str, budget: float, risk_tolerance: str) -> dict:
        self._orchestrator.decision.budget         = budget
        self._orchestrator.decision.risk_tolerance = risk_tolerance
        return await self._orchestrator.run_pipeline(trigger=trigger)

    def get_status(self) -> dict:
        return self._orchestrator.status_report()

    def get_log(self, limit: int = 50) -> list[dict]:
        return self._orchestrator.get_full_log()[-limit:]


agent_service = AgentService()
