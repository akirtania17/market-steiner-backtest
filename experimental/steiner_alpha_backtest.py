# steiner_alpha_backtest.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Config
# =============================================================================

# <<< CHANGE THIS IF YOUR FILE NAME IS DIFFERENT >>>
DATA_PATH = "data/real_etf_fut_5m.csv"  # or "real_etf_fut_data.csv"

INITIAL_CAPITAL = 1_000_000.0

# Risk / sizing
RISK_TARGET_ANNUAL = 0.10      # target annual vol of the strategy (10%)
MAX_LEVERAGE = 1.0             # max |pair notional| / capital
ZSCORE_ENTRY_SCALE = 1.0       # we scale position ~ zscore, but squash with tanh

# Vol windows
SPREAD_MEAN_WINDOW = 60        # bars for rolling mean (e.g. 60 * 5min = 5h)
SPREAD_VOL_WINDOW = 60         # bars for rolling std

# Costs (simple but realistic-ish)
ETF_HALF_SPREAD_BPS = 0.5      # half-spread in bps
FUT_HALF_SPREAD_BPS = 0.25     # futures half-spread
COMMISSION_PER_NOTIONAL_BPS = 0.01  # 0.01bp commission just as placeholder

# Trading calendar assumptions
MINUTES_PER_DAY = 390          # US cash session
DAYS_PER_YEAR = 252
BARS_PER_DAY = MINUTES_PER_DAY // 5  # if 5m bars → 78


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    final_equity: float
    total_pnl: float
    sharpe_annualized: float
    turnover_notional_per_capital: float
    max_drawdown: float


def load_etf_fut_data(path: str) -> pd.DataFrame:
    """
    Load intraday ETF/FUT mid-price data produced by make_real_data.py.

    Expected columns:
        timestamp, etf_mid, fut_mid
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # Try to auto-detect column names if user changed them
    cols = {c.lower(): c for c in df.columns}
    etf_col = cols.get("etf_mid") or cols.get("etfclose") or cols.get("etf")
    fut_col = cols.get("fut_mid") or cols.get("futclose") or cols.get("future")

    if etf_col is None or fut_col is None:
        raise RuntimeError(
            f"Could not find ETF/FUT columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[etf_col, fut_col]].rename(columns={etf_col: "etf_mid",
                                                fut_col: "fut_mid"})

    # Ensure strictly increasing timestamps, drop duplicates
    df = df[~df.index.duplicated()].sort_index()

    # Resample to a regular 5-minute grid and ffill prices so ETF & FUT
    # are aligned in time.
    df = (
        df.resample("5min")
        .last()
        .ffill()
        .dropna(subset=["etf_mid", "fut_mid"])
    )

    return df


def compute_spread_and_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the "Steiner spread" and signal.

    Here we just use log-price spread and z-score it. You can later swap in
    a fancier Steiner-based hedge ratio; everything downstream will still work.
    """
    # Log prices
    df["log_etf"] = np.log(df["etf_mid"])
    df["log_fut"] = np.log(df["fut_mid"])

    # Assume long ETF / short FUT as the basic pair; spread = ETF - FUT
    df["spread"] = df["log_etf"] - df["log_fut"]

    # Rolling mean & vol for z-score
    df["spread_mean"] = df["spread"].rolling(SPREAD_MEAN_WINDOW, min_periods=SPREAD_MEAN_WINDOW // 2).mean()
    df["spread_std"] = df["spread"].rolling(SPREAD_VOL_WINDOW, min_periods=SPREAD_VOL_WINDOW // 2).std()

    # Z-score; where vol is tiny, set z = 0 to avoid nonsense
    df["zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    df.loc[df["spread_std"] <= 1e-8, "zscore"] = 0.0

    # Smooth the signal slightly (optional)
    df["signal_raw"] = df["zscore"]
    df["signal"] = np.tanh(df["signal_raw"] / ZSCORE_ENTRY_SCALE)

    return df


def position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility-targeted position sizing with leverage cap.

    We treat 'notional_pair' as exposure to the spread:
        pnl_t ≈ notional_pair_{t-1} * Δspread_t
    """
    df = df.copy()

    # Realized bar-to-bar spread vol for risk targeting
    df["spread_ret"] = df["spread"].diff()
    # Use rolling std of returns as per-bar volatility
    df["spread_ret_vol"] = (
        df["spread_ret"].rolling(SPREAD_VOL_WINDOW, min_periods=SPREAD_VOL_WINDOW // 2).std()
    )

    # Convert annual risk target to per-bar volatility target
    # Annual var ≈ (daily_vol * sqrt(252))^2 → daily_vol_target ≈ RISK_TARGET_ANNUAL / sqrt(252)
    daily_vol_target = RISK_TARGET_ANNUAL / math.sqrt(DAYS_PER_YEAR)
    bar_vol_target = daily_vol_target / math.sqrt(BARS_PER_DAY)

    capital = INITIAL_CAPITAL

    notional_pair = []
    trade_costs = []
    delta_notional = []

    prev_notional = 0.0

    for ts, row in df.iterrows():
        sig = float(row.get("signal", 0.0))
        bar_vol = float(row.get("spread_ret_vol", np.nan))

        if np.isnan(bar_vol) or bar_vol < 1e-8:
            target_notional = 0.0
        else:
            # position ∝ signal * (vol_target / realized_vol)
            # so that exposure shrinks when realized vol is high
            scale = bar_vol_target / bar_vol
            target_notional = sig * scale * capital

        # Hard leverage cap: |notional_pair| ≤ MAX_LEVERAGE * capital
        cap_notional = MAX_LEVERAGE * capital
        target_notional = float(np.clip(target_notional, -cap_notional, cap_notional))

        d_notional = target_notional - prev_notional

        # Trading cost: proportional to turnover and average half-spread
        avg_half_spread_bps = (ETF_HALF_SPREAD_BPS + FUT_HALF_SPREAD_BPS) / 2.0
        total_cost_bps = avg_half_spread_bps + COMMISSION_PER_NOTIONAL_BPS

        trade_cost = abs(d_notional) * (total_cost_bps / 10_000.0)

        notional_pair.append(target_notional)
        delta_notional.append(d_notional)
        trade_costs.append(trade_cost)

        prev_notional = target_notional

    df["notional_pair"] = notional_pair
    df["delta_notional_pair"] = delta_notional
    df["trade_cost"] = trade_costs

    return df


def run_backtest(df_raw: pd.DataFrame) -> BacktestResult:
    """
    Main backtest loop.
    """
    df = df_raw.copy()

    df = compute_spread_and_signal(df)
    df = position_sizing(df)

    # PnL: exposure to spread changes minus trading costs
    df["spread_ret"] = df["spread"].diff().fillna(0.0)

    # Shift notional for PnL (use previous bar's exposure)
    df["prev_notional_pair"] = df["notional_pair"].shift(1).fillna(0.0)

    df["gross_pnl"] = df["prev_notional_pair"] * df["spread_ret"]
    df["net_pnl"] = df["gross_pnl"] - df["trade_cost"]

    df["equity"] = INITIAL_CAPITAL + df["net_pnl"].cumsum()

    # Risk stats
    returns = df["equity"].pct_change().dropna()
    if len(returns) > 2:
        sharpe_ann = (
            returns.mean()
            / (returns.std() + 1e-12)
            * math.sqrt(DAYS_PER_YEAR * BARS_PER_DAY)
        )
    else:
        sharpe_ann = float("nan")

    total_pnl = df["equity"].iloc[-1] - INITIAL_CAPITAL
    final_equity = df["equity"].iloc[-1]

    # Turnover
    total_turnover = df["delta_notional_pair"].abs().sum()
    turnover_per_capital = total_turnover / INITIAL_CAPITAL

    # Max drawdown
    roll_max = df["equity"].cummax()
    drawdown = df["equity"] / roll_max - 1.0
    max_dd = drawdown.min()

    # Sample trades DataFrame (filter where we actually change exposure)
    trades = df.loc[df["delta_notional_pair"].abs() > 1e-6, [
        "prev_notional_pair",
        "notional_pair",
        "delta_notional_pair",
        "trade_cost",
        "signal",
    ]].copy()

    trades = trades.rename(columns={"notional_pair": "new_notional_pair"})

    return BacktestResult(
        equity=df["equity"],
        trades=trades,
        final_equity=float(final_equity),
        total_pnl=float(total_pnl),
        sharpe_annualized=float(sharpe_ann),
        turnover_notional_per_capital=float(turnover_per_capital),
        max_drawdown=float(max_dd),
    )


def plot_equity_curve(equity: pd.Series) -> None:
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Steiner ETF–FUT Alpha Equity Curve")
    plt.xlabel("")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = load_etf_fut_data(DATA_PATH)

    print(f"Data shape after cleaning: {df.shape}")
    print(df.head())

    result = run_backtest(df)

    print("\n=== Backtest summary ===")
    print(f"final_equity: {result.final_equity:,.4f}")
    print(f"total_pnl: {result.total_pnl:,.4f}")
    print(f"sharpe_annualized: {result.sharpe_annualized: .4f}")
    print(
        f"turnover_notional_per_capital: "
        f"{result.turnover_notional_per_capital: .4f}"
    )
    print(f"max_drawdown: {result.max_drawdown: .4f}")

    print("\nSample trades:")
    print(result.trades.head())

    plot_equity_curve(result.equity)


if __name__ == "__main__":
    main()
