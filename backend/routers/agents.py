"""
Agent management endpoints + WebSocket for live agent log streaming.
"""

import asyncio
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.core.security import get_current_user
from backend.models.schemas import PipelineRunRequest
from backend.services.agent_service import agent_service

router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.get("/status", summary="Get status of all agents")
def get_status(_: dict = Depends(get_current_user)):
    """Returns current status (idle/running/alert/error) for each agent."""
    return agent_service.get_status()


@router.get("/log", summary="Get recent agent decision log")
def get_log(limit: int = 50, _: dict = Depends(get_current_user)):
    """Returns the last `limit` entries from the combined agent log."""
    return agent_service.get_log(limit=limit)


@router.post("/run", summary="Manually trigger the investment pipeline")
async def run_pipeline(body: PipelineRunRequest, _: dict = Depends(get_current_user)):
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
async def websocket_log(websocket: WebSocket, token: str = ""):
    """
    WebSocket endpoint — streams agent log entries in real time.
    Pass JWT as a query parameter: /ws/log?token=<bearer-token>
    """
    from backend.core.security import decode_token
    from fastapi import status as http_status
    try:
        decode_token(token)
    except Exception:
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return

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
