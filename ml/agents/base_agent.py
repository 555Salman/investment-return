"""
Base class shared by all agents.
Each agent has a name, a status, and an async run() method.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    ALERT   = "alert"
    ERROR   = "error"


class BaseAgent:
    def __init__(self, name: str):
        self.name   = name
        self.status = AgentStatus.IDLE
        self.log: list[dict] = []

    def _record(self, action: str, detail: dict | None = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent":     self.name,
            "action":    action,
            "detail":    detail or {},
        }
        self.log.append(entry)
        logger.info(f"[{self.name}] {action}  {detail or ''}")
        return entry

    async def run(self, *args, **kwargs):
        raise NotImplementedError
