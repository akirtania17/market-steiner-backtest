from __future__ import annotations

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Compute max drawdown from an equity curve.
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_annualized_sharpe(
    pnl_series: pd.Series,
    bars_per_year: int,
) -> float:
    """
    Compute annualized Sharpe ratio from per-bar PnL.
    """
    pnl = pnl_series.to_numpy()
    if pnl.size < 2:
        return 0.0
    mean = pnl.mean()
    std = pnl.std(ddof=1)
    if std <= 0:
        return 0.0
    sharpe = (mean / std) * np.sqrt(bars_per_year)
    return float(sharpe)


def compute_turnover(
    trades_notional: pd.Series,
    initial_capital: float,
) -> float:
    """
    Turnover = sum(|Δnotional|) / initial_capital
    """
    total_traded = trades_notional.abs().sum()
    if initial_capital <= 0:
        return float("nan")
    return float(total_traded / initial_capital)
