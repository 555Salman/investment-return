"""
Forecast service — wraps ForecastAgent for use inside FastAPI.
Singleton pattern so models are loaded once at startup.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.forecast_agent import ForecastAgent


class ForecastService:
    def __init__(self):
        self._agent = ForecastAgent()

    async def get_all_forecasts(self) -> dict:
        forecasts = await self._agent.run()
        return {
            "forecasts": {
                pair: {
                    **data,
                    "forecast_return_pct": f"{data['forecast_return']*100:.3f}%",
                }
                for pair, data in forecasts.items()
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_pair_forecast(self, pair: str) -> dict | None:
        self._agent._load_models()
        result = self._agent.forecast_pair(pair)
        if result:
            result["forecast_return_pct"] = f"{result['forecast_return']*100:.3f}%"
        return result


forecast_service = ForecastService()
