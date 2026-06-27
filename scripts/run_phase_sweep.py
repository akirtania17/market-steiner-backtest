# scripts/run_phase_sweep.py
from __future__ import annotations

import os
import sys
import pprint

import pandas as pd

# --- Ensure src/ is on sys.path so "market_steiner" can be imported ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_steiner.config import BacktestConfig
from market_steiner.backtest.param_sweep import sweep_phase


def main():
    base_cfg = BacktestConfig()

    # Grids centered around your current "sweet spot" config:
    #
    #   lambda_etf ≈ 0.02
    #   lambda_fut ≈ 0.05
    #   alpha      ≈ 0.02
    #
    lambda_etf_grid = [0.01, 0.02, 0.05]
    lambda_fut_grid = [0.02, 0.05, 0.08]
    alpha_grid = [0.005, 0.01, 0.02, 0.05]

    print("Running phase sweep...")
    df = sweep_phase(
        base_cfg=base_cfg,
        lambda_etf_grid=lambda_etf_grid,
        lambda_fut_grid=lambda_fut_grid,
        alpha_grid=alpha_grid,
    )

    # Save full results
    out_path = os.path.join(PROJECT_ROOT, "phase_sweep_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full phase sweep results to: {out_path}")

    # Show a small preview
    print("\n=== Head of results ===")
    print(df.head())

    # Show top configs by Sharpe
    top_by_sharpe = df.sort_values("sharpe_annualized", ascending=False).head(10)
    print("\n=== Top 10 configs by Sharpe ===")
    pprint.pp(top_by_sharpe.to_dict(orient="records"))

    # Optional: also show how activation fractions move
    print("\n=== Example rows with mixed activation (ETF_ONLY & FUT_ONLY both used) ===")
    mixed = df[
        (df["frac_etf_only"] > 0.1)
        & (df["frac_fut_only"] > 0.1)
    ].sort_values("sharpe_annualized", ascending=False).head(10)
    if not mixed.empty:
        pprint.pp(mixed.to_dict(orient="records"))
    else:
        print("No strongly mixed-activation configs found in this sweep.")


if __name__ == "__main__":
    main()
