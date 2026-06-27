from __future__ import annotations

import os
import sys
import pprint

# --- Ensure src/ is on sys.path so "market_steiner" can be imported ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Now imports from the package will work ---

from market_steiner.config import BacktestConfig
from market_steiner.backtest.engine import run_single_backtest


def main():
    # Default config; customize as desired.
    cfg = BacktestConfig()

    result = run_single_backtest(cfg)

    print("\n=== Backtest Summary ===")
    pprint.pprint(result.summary, sort_dicts=False)
    
    # Show first 15 rows of logs to sanity-check behavior
    print("\n=== First 15 log rows ===")
    print(result.logs.head(15))

    print("\n=== Activation counts ===")
    print(result.logs["activation"].value_counts())

    # If you want to save logs:
    # result.logs.to_csv("steiner_hedge_logs.csv")


if __name__ == "__main__":
    main()
