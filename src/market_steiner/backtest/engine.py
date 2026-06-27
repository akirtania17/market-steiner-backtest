from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from ..config import BacktestConfig
from ..data.loader import load_etf_fut_csv
from ..data.preprocessing import align_and_clean, compute_log_returns, merge_prices_and_returns
from ..strategy.steiner_hedge import SteinerHedgeStrategy
from .metrics import compute_max_drawdown, compute_annualized_sharpe, compute_turnover
from ..strategy.simple_hedge import SimpleHedgeStrategy


@dataclass
class BacktestResult:
    config: BacktestConfig
    logs: pd.DataFrame
    summary: Dict[str, Any]


def run_single_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a single ETF–FUT Steiner hedge backtest with the given configuration.
    """
    # 1. Load & preprocess data
    prices = load_etf_fut_csv(config.data)
    prices = align_and_clean(prices)
    rets = compute_log_returns(prices)
    df = merge_prices_and_returns(prices, rets)

    # 2. Initialize strategy
    strat = SteinerHedgeStrategy(config)

    # 3. Run strategy
    logs = strat.run_on_dataframe(df)

    # 4. Compute metrics
    equity = logs["equity"]
    pnl_gross = logs["pnl_gross"]
    costs = logs["cost_total"]

    pnl_net = pnl_gross - costs
    pnl_net.name = "pnl_net"

    max_dd = compute_max_drawdown(equity)
    sharpe = compute_annualized_sharpe(
        pnl_series=pnl_net,
        bars_per_year=config.bars_per_year,
    )

    # Notional turnover: sum of absolute ETF+FUT trades / capital
    trades_notional = logs["trade_etf"].abs() + logs["trade_fut"].abs()
    turnover = compute_turnover(
        trades_notional=trades_notional,
        initial_capital=config.costs.initial_capital,
    )

    summary = {
        "final_equity": float(equity.iloc[-1]),
        "total_pnl_gross": float(pnl_gross.sum()),
        "total_costs": float(costs.sum()),
        "total_pnl_net": float(pnl_net.sum()),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "turnover_notional_per_capital": turnover,
        "n_bars": int(len(logs)),
    }

    return BacktestResult(
        config=config,
        logs=logs,
        summary=summary,
    )

@dataclass
class BaselineResult:
    config: BacktestConfig
    logs: pd.DataFrame
    summary: Dict[str, Any]


def run_simple_baseline(config: BacktestConfig) -> BaselineResult:
    """
    Run the non-Steiner baseline hedge using both ETF and FUT with no activation.
    """
    # 1. Load & preprocess data (same as run_single_backtest)
    prices = load_etf_fut_csv(config.data)
    prices = align_and_clean(prices)
    rets = compute_log_returns(prices)
    df = merge_prices_and_returns(prices, rets)

    # 2. Initialize baseline strategy
    strat = SimpleHedgeStrategy(config)

    # 3. Run strategy
    logs = strat.run_on_dataframe(df)

    # 4. Compute metrics (same as Steiner)
    equity = logs["equity"]
    pnl_gross = logs["pnl_gross"]
    costs = logs["cost_total"]

    pnl_net = pnl_gross - costs
    pnl_net.name = "pnl_net"

    max_dd = compute_max_drawdown(equity)
    sharpe = compute_annualized_sharpe(
        pnl_series=pnl_net,
        bars_per_year=config.bars_per_year,
    )

    trades_notional = logs["trade_etf"].abs() + logs["trade_fut"].abs()
    turnover = compute_turnover(
        trades_notional=trades_notional,
        initial_capital=config.costs.initial_capital,
    )

    summary = {
        "final_equity": float(equity.iloc[-1]),
        "total_pnl_gross": float(pnl_gross.sum()),
        "total_costs": float(costs.sum()),
        "total_pnl_net": float(pnl_net.sum()),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "turnover_notional_per_capital": turnover,
        "n_bars": int(len(logs)),
    }

    return BaselineResult(
        config=config,
        logs=logs,
        summary=summary,
    )
