from __future__ import annotations

import os
import sys
import pprint

# --- Ensure src/ is on sys.path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_steiner.config import BacktestConfig
from market_steiner.backtest.engine import (
    run_single_backtest,
    run_simple_baseline,
)


def main():
    cfg = BacktestConfig()

    print("\n=== Running Steiner Hedge ===")
    steiner_result = run_single_backtest(cfg)
    pprint.pprint(steiner_result.summary, sort_dicts=False)

    print("\n=== Running Simple Baseline Hedge (no Steiner) ===")
    baseline_result = run_simple_baseline(cfg)
    pprint.pprint(baseline_result.summary, sort_dicts=False)

    # Optional: compare key metrics
    print("\n=== Comparison ===")
    print(f"Steiner Sharpe:  {steiner_result.summary['sharpe_annualized']:.4f}")
    print(f"Baseline Sharpe: {baseline_result.summary['sharpe_annualized']:.4f}")
    print(f"Steiner Turnover:  {steiner_result.summary['turnover_notional_per_capital']:.4f}")
    print(f"Baseline Turnover: {baseline_result.summary['turnover_notional_per_capital']:.4f}")


if __name__ == "__main__":
    main()
