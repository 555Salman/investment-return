"""
Agent management endpoints + WebSocket for live agent log streaming.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.models.schemas import PipelineRunRequest
from backend.services.agent_service import agent_service

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.get("/status", summary="Get status of all agents")
def get_status():
    """Returns current status (idle/running/alert/error) for each agent."""
    return agent_service.get_status()


@router.get("/log", summary="Get recent agent decision log")
def get_log(limit: int = 50):
    """Returns the last `limit` entries from the combined agent log."""
    return agent_service.get_log(limit=limit)


@router.post("/run", summary="Manually trigger the investment pipeline")
async def run_pipeline(body: PipelineRunRequest):
    """
    Triggers one full pipeline cycle:
    Forecast → Decision → Rebalance.
    Returns a summary of actions taken.
    """
    return await agent_service.run_pipeline(
        trigger=body.trigger,
        budget=body.budget,
        risk_tolerance=body.risk_tolerance,
    )


@router.websocket("/ws/log")
async def websocket_log(websocket: WebSocket):
    """
    WebSocket endpoint — streams agent log entries in real time.
    The frontend connects here to receive live agent activity updates.
    """
    await websocket.accept()
    seen = 0
    try:
        while True:
            log = agent_service.get_log(limit=200)
            new_entries = log[seen:]
            for entry in new_entries:
                await websocket.send_json(entry)
            seen += len(new_entries)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
