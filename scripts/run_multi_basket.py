# scripts/run_multi_basket.py
from __future__ import annotations

import os
import sys
import pprint

import numpy as np
import pandas as pd

# --- Ensure project root is on sys.path so "market_steiner" imports work ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_steiner.config import (
    BacktestConfig,
    BasketConfig,
    BasketInstrumentConfig,
)
from market_steiner.strategy.multi_basket import MultiEtfSteinerStrategy


# ---------- Simple metrics helper (inline) ----------

def compute_simple_metrics(equity: pd.Series) -> dict:
    """
    Compute basic backtest metrics from an equity time series.
    Returns:
      - final_equity
      - total_pnl
      - sharpe_annualized
      - max_drawdown
      - n_bars
    """
    eq = pd.Series(equity).astype(float)
    eq = eq.dropna()
    n = len(eq)

    if n < 2:
        return {
            "final_equity": float(eq.iloc[-1]) if n > 0 else np.nan,
            "total_pnl": np.nan,
            "sharpe_annualized": 0.0,
            "max_drawdown": np.nan,
            "n_bars": n,
        }

    pnl = float(eq.iloc[-1] - eq.iloc[0])
    rets = eq.diff().dropna() / eq.shift(1).dropna()

    if len(rets) > 1 and rets.std(ddof=1) > 0:
        # 252 trading days * 78 5-min bars/day
        ann_factor = np.sqrt(252.0 * 78.0)
        sharpe = float(rets.mean() / rets.std(ddof=1) * ann_factor)
    else:
        sharpe = 0.0

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min())

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_pnl": pnl,
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "n_bars": n,
    }


# ---------- Data loader ----------

def load_multi_basket_data(csv_path: str) -> pd.DataFrame:
    """
    Load a multi-ETF/FUT intraday dataset.

    Assumes a clean CSV with columns:
      timestamp, qqq_mid, xlk_mid, nq_mid, qqq_ret, xlk_ret, nq_ret

    If your file already has timestamp as index, we handle that too.
    """
    df = pd.read_csv(csv_path)

    # If there is a 'timestamp' column, use it; otherwise assume index-like first column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        # Fallback: first column is timestamp-like
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df.index.name = "timestamp"
    
    # 🔹 Drop any NaT index rows (like that extra header/blank row)
    df = df[~df.index.isna()]

    # Ensure numeric columns
    for col in ["qqq_mid", "xlk_mid", "nq_mid", "qqq_ret", "xlk_ret", "nq_ret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN returns with 0 for safety
    for col in ["qqq_ret", "xlk_ret", "nq_ret"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


# ---------- Main ----------

def main():
    cfg = BacktestConfig()

    csv_path = os.path.join(PROJECT_ROOT, "data", "multi_etf_fut_5m.csv")
    print(f"Loading data from {csv_path} ...")
    df = load_multi_basket_data(csv_path)
    print(f"Data shape: {df.shape}")
    print(df.head())

    # ---- Basket Config: hedge XLK using QQQ + NQ ----
    #
    # target_ret_col: what we are trying to hedge (here: XLK)
    # instruments: which instruments the Steiner tree can route through
    #
    basket_cfg = BasketConfig(
        target_ret_col="xlk_ret",
        instruments=[
            BasketInstrumentConfig(
                name="QQQ", price_col="qqq_mid", ret_col="qqq_ret", lambda_leak=0.05
            ),
            BasketInstrumentConfig(
                name="NQ", price_col="nq_mid", ret_col="nq_ret", lambda_leak=0.02
            ),
        ],
    )

    strat = MultiEtfSteinerStrategy(cfg=cfg, basket_cfg=basket_cfg)
    log_df = strat.run_on_dataframe(df)

    print("\n=== First 10 log rows ===")
    print(log_df.head(10))

    equity = log_df["equity"]
    metrics = compute_simple_metrics(equity)

    print("\n=== Multi-basket Steiner Backtest Summary ===")
    pprint.pp(metrics)

    # Activation statistics
    active_mask_df = pd.DataFrame(log_df["active_mask"].tolist(), index=log_df.index)
    print("\n=== Activation Fractions ===")
    for col in active_mask_df.columns:
        frac_active = active_mask_df[col].mean()
        print(f"{col}: {frac_active:.3f}")


if __name__ == "__main__":
    main()
