"""
Market Monitor Agent.
Polls the latest exchange rate for each currency pair and raises an alert
when a significant price shift is detected (default threshold: 0.5%).
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.base_agent import BaseAgent, AgentStatus

TICKERS = {
    "EUR_USD": "EURUSD=X",
    "AUD_USD": "AUDUSD=X",
    "NZD_USD": "NZDUSD=X",
}


class MarketMonitorAgent(BaseAgent):
    """
    Continuously polls live exchange rates and emits alerts when
    price moves exceed the configured threshold.

    Args:
        threshold:       Minimum absolute log-return to trigger an alert (e.g. 0.005 = 0.5%)
        poll_interval:   Seconds between polls (default 60s; use shorter for testing)
    """

    def __init__(self, threshold: float = 0.005, poll_interval: int = 60):
        super().__init__("MarketMonitor")
        self.threshold     = threshold
        self.poll_interval = poll_interval
        self.last_prices: dict[str, float] = {}
        self.alerts: list[dict] = []

    def fetch_latest(self) -> dict[str, float]:
        """Fetch the most recent close price for each pair."""
        prices = {}
        for name, ticker in TICKERS.items():
            try:
                data = yf.download(ticker, period="2d", interval="1d",
                                   auto_adjust=True, progress=False)
                if not data.empty:
                    prices[name] = float(data["Close"].iloc[-1])
            except Exception as exc:
                self.status = AgentStatus.ERROR
                self._record("fetch_error", {"pair": name, "error": str(exc)})
        return prices

    def detect_shifts(self, current: dict[str, float]) -> list[dict]:
        """Compare current prices against last known prices and flag big moves."""
        alerts = []
        for pair, price in current.items():
            if pair in self.last_prices:
                log_ret = np.log(price / self.last_prices[pair])
                if abs(log_ret) >= self.threshold:
                    alert = {
                        "pair":      pair,
                        "log_return": round(log_ret, 6),
                        "prev_price": self.last_prices[pair],
                        "curr_price": price,
                        "direction":  "up" if log_ret > 0 else "down",
                    }
                    alerts.append(alert)
                    self.alerts.append(alert)
                    self._record("ALERT", alert)
                    self.status = AgentStatus.ALERT
        return alerts

    async def run(self, max_polls: int | None = None) -> None:
        """
        Poll loop. Runs indefinitely (or up to max_polls for testing).
        """
        self.status = AgentStatus.RUNNING
        self._record("started", {"threshold": self.threshold, "interval": self.poll_interval})
        polls = 0

        while True:
            current = self.fetch_latest()
            if current:
                alerts = self.detect_shifts(current)
                self.last_prices.update(current)
                if not alerts:
                    self.status = AgentStatus.RUNNING
                    self._record("poll_ok", {"prices": current})

            polls += 1
            if max_polls is not None and polls >= max_polls:
                break

            await asyncio.sleep(self.poll_interval)

        self.status = AgentStatus.IDLE
        self._record("stopped")
