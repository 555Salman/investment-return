"""
Hyperparameter tuning for the LSTM model.

Two strategies:
  1. Bayesian Optimisation via Optuna  (recommended)
  2. Grid Search                        (exhaustive, slower)

Usage:
    python ml/training/hyperparameter_tuning.py --pair USD_EUR --strategy bayesian --trials 30
    python ml/training/hyperparameter_tuning.py --pair USD_EUR --strategy grid
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ml.training.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "tuning_results"


# ── Bayesian Optimisation ──────────────────────────────────────────────────────

def bayesian_search(pair_name: str, n_trials: int = 30) -> dict:
    """
    Use Optuna to search for the best hyperparameters.
    Minimises validation loss (MSE) over n_trials.
    """
    logger.info(f"Bayesian search: {n_trials} trials for {pair_name}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "pair_name":   pair_name,
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":  trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs":      50,
            "patience":    10,
        }
        result = train(**params)
        return result["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["strategy"] = "bayesian"
    best["best_val_loss"] = study.best_value
    logger.info(f"Best params: {best}")
    _save_results(pair_name, "bayesian", best)
    return best


# ── Grid Search ────────────────────────────────────────────────────────────────

GRID = {
    "hidden_size": [64, 128],
    "num_layers":  [1, 2],
    "dropout":     [0.2, 0.3],
    "lr":          [1e-3, 5e-4],
    "batch_size":  [64],
}

def grid_search(pair_name: str) -> dict:
    """
    Exhaustive grid search over GRID. Trains one model per combination.
    """
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    logger.info(f"Grid search: {len(combos)} combinations for {pair_name}")

    best_loss   = float("inf")
    best_params = {}

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        logger.info(f"  [{i}/{len(combos)}] {params}")
        result = train(
            pair_name=pair_name,
            epochs=30,
            patience=8,
            **params,
        )
        if result["best_val_loss"] < best_loss:
            best_loss   = result["best_val_loss"]
            best_params = params.copy()
            best_params["best_val_loss"] = best_loss

    best_params["strategy"] = "grid"
    logger.info(f"Best params: {best_params}")
    _save_results(pair_name, "grid", best_params)
    return best_params


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_results(pair_name: str, strategy: str, params: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{pair_name}_{strategy}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved results → {path}")


def load_best_params(pair_name: str, strategy: str = "bayesian") -> dict:
    path = RESULTS_DIR / f"{pair_name}_{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(f"No tuning results at {path}. Run tuning first.")
    with open(path) as f:
        return json.load(f)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",     default="USD_EUR")
    parser.add_argument("--strategy", default="bayesian", choices=["bayesian", "grid"])
    parser.add_argument("--trials",   default=30, type=int, help="Optuna trials (bayesian only)")
    args = parser.parse_args()

    if args.strategy == "bayesian":
        bayesian_search(args.pair, n_trials=args.trials)
    else:
        grid_search(args.pair)
