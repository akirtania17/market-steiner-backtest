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
from market_steiner.backtest.param_sweep import sweep_lambda_and_k


def main():
    base_cfg = BacktestConfig()

    lambda_etf_grid = [0.05, 0.1, 0.2]
    lambda_fut_grid = [0.01, 0.02]
    k_l2_grid = [1e-4, 1e-3, 1e-2]

    df = sweep_lambda_and_k(
        base_cfg=base_cfg,
        lambda_etf_grid=lambda_etf_grid,
        lambda_fut_grid=lambda_fut_grid,
        k_l2_grid=k_l2_grid,
    )

    print("\n=== Parameter Sweep Results (head) ===")
    print(df.head())

    # You can save to CSV:
    # df.to_csv("steiner_param_sweep_results.csv", index=False)

    # Or print the top configs by Sharpe:
    top = df.sort_values("sharpe_annualized", ascending=False).head(10)
    print("\n=== Top Configs by Sharpe ===")
    pprint.pp(top.to_dict(orient="records"))


if __name__ == "__main__":
    main()
