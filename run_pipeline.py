"""
Full pipeline runner — executes all five project phases in sequence:

  Phase 1: Data preprocessing (cleaning, feature engineering, sequences)
  Phase 2: LSTM training (one model per currency pair, with early stopping)
  Phase 3: Walk-forward backtesting (LP vs equal-weight vs buy-and-hold)
  Phase 4: Portfolio optimisation snapshot (LP on latest forecasts)
  Phase 5: Agentic AI dry run (one pipeline cycle in simulation mode)

Usage:
    python run_pipeline.py                     # all phases
    python run_pipeline.py --skip-agents       # skip the async agent run
    python run_pipeline.py --phases 1 2        # run only phases 1 and 2
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PAIRS = ["EUR_USD", "AUD_USD", "NZD_USD"]


# ── Phase 1: Preprocessing ─────────────────────────────────────────────────────

def phase1_preprocess() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1 — Data Preprocessing")
    logger.info("=" * 60)

    from ml.data_pipeline.ingestion import load_all
    from ml.data_pipeline.preprocessing import run_pipeline

    data = load_all()
    if not data:
        logger.error("No raw data found. Ensure data/raw/ contains CSV files.")
        return

    for name, df in data.items():
        run_pipeline(df, pair_name=name, sequence_length=60)

    logger.info("Phase 1 complete.\n")


# ── Phase 2: Training ──────────────────────────────────────────────────────────

def phase2_train(epochs: int = 50, patience: int = 10) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 2 — LSTM Training")
    logger.info("=" * 60)

    from ml.training.train import train

    results = {}
    for pair in PAIRS:
        try:
            logger.info(f"\nTraining {pair} ...")
            results[pair] = train(
                pair_name=pair,
                epochs=epochs,
                patience=patience,
                seed=42,
            )
        except FileNotFoundError:
            logger.warning(
                f"  Preprocessed data for {pair} not found — skipping. Run Phase 1 first."
            )

    logger.info("\nPhase 2 complete.\n")
    return results


# ── Phase 3: Backtesting ───────────────────────────────────────────────────────

def phase3_backtest() -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 3 — Walk-Forward Backtesting")
    logger.info("=" * 60)

    from ml.training.backtest import run_backtest

    try:
        results = run_backtest(
            budget=10_000.0,
            risk_tolerance="medium",
            rebalance_every=1,
        )
    except RuntimeError as exc:
        logger.error(f"Backtesting failed: {exc}")
        return {}

    logger.info("\nPhase 3 complete.\n")
    return results


# ── Phase 4: Portfolio optimisation ───────────────────────────────────────────

def phase4_optimise() -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 4 — Portfolio Optimisation (latest forecasts)")
    logger.info("=" * 60)

    import numpy as np
    import torch

    from ml.models.lstm_model import LSTMForecaster
    from ml.models.model_utils import load_model
    from ml.optimization.portfolio_optimizer import compare_strategies

    PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
    forecast_returns: dict[str, float] = {}

    for pair in PAIRS:
        base = PROCESSED_DIR / pair
        try:
            X = np.load(base / "X_test.npy")
            y = np.load(base / "y_test.npy")
            m = LSTMForecaster(input_size=X.shape[2])
            m = load_model(m, pair, tag="best")
            m.eval()
            with torch.no_grad():
                pred = float(m(torch.tensor(X[-1:], dtype=torch.float32)).item())
            last = float(y[-1])
            forecast_returns[pair] = (pred - last) / (last + 1e-8)
        except FileNotFoundError:
            logger.warning(f"  Model or data for {pair} not found — skipping.")

    if not forecast_returns:
        logger.error("No forecasts available; train models first (Phase 2).")
        return {}

    strategies = compare_strategies(
        forecast_returns,
        budget=10_000.0,
        risk_tolerance="medium",
    )

    logger.info("\nOptimisation results:")
    for s in strategies:
        logger.info(
            f"  {s['strategy']:<15}  "
            f"expected return: {s['expected_return_pct']*100:.2f}%  "
            f"amount: ${s['expected_amount']:,.2f}"
        )

    logger.info("\nPhase 4 complete.\n")
    return {"strategies": strategies, "forecast_returns": forecast_returns}


# ── Phase 5: Agentic AI dry run ────────────────────────────────────────────────

async def phase5_agents() -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 5 — Agentic AI Pipeline (simulation mode)")
    logger.info("=" * 60)

    from ml.agents.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        budget=10_000.0,
        risk_tolerance="medium",
        simulation=True,
    )

    try:
        summary = await orchestrator.run_pipeline(trigger="manual")
        logger.info(f"\nAgent pipeline summary: {summary}")
    except Exception as exc:
        logger.error(f"Agent pipeline failed: {exc}")
        return {}

    logger.info("\nPhase 5 complete.\n")
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full multicurrency investment pipeline."
    )
    parser.add_argument(
        "--phases", nargs="*", type=int, default=[1, 2, 3, 4, 5],
        metavar="N", help="Phases to run (default: all 1-5)",
    )
    parser.add_argument(
        "--skip-agents", action="store_true",
        help="Skip Phase 5 (agentic AI run)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for Phase 2 (default: 50)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience for Phase 2 (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    phases = set(args.phases)
    if args.skip_agents:
        phases.discard(5)

    if 1 in phases:
        phase1_preprocess()
    if 2 in phases:
        phase2_train(epochs=args.epochs, patience=args.patience)
    if 3 in phases:
        phase3_backtest()
    if 4 in phases:
        phase4_optimise()
    if 5 in phases:
        asyncio.run(phase5_agents())

    logger.info("All selected phases complete.")
