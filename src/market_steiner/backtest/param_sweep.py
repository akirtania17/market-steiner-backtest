# src/market_steiner/backtest/param_sweep.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import pandas as pd

from ..config import BacktestConfig
from .engine import run_single_backtest


@dataclass
class SweepResultRow:
    lambda_etf: float
    lambda_fut: float
    alpha: float
    sharpe_annualized: float
    turnover_notional_per_capital: float
    total_pnl_net: float
    total_costs: float
    frac_etf_only: float
    frac_fut_only: float
    frac_both: float
    frac_none: float


def sweep_phase(
    base_cfg: BacktestConfig,
    lambda_etf_grid: Iterable[float],
    lambda_fut_grid: Iterable[float],
    alpha_grid: Iterable[float],
) -> pd.DataFrame:
    """
    Sweep over (lambda_etf, lambda_fut, alpha) and measure:

      - Sharpe
      - Turnover
      - Net PnL
      - Costs
      - Activation frequencies (ETF_ONLY, FUT_ONLY, BOTH, NONE)

    ETF amplification is whatever you currently have set inside compute_raw_hedge.
    """
    rows: List[Dict[str, Any]] = []

    for lam_etf in lambda_etf_grid:
        for lam_fut in lambda_fut_grid:
            for alpha in alpha_grid:
                cfg = deepcopy(base_cfg)
                cfg.steiner.lambda_etf = lam_etf
                cfg.steiner.lambda_fut = lam_fut
                cfg.steiner.alpha = alpha

                result = run_single_backtest(cfg)

                summary = result.summary
                logs = result.logs

                counts = logs["activation"].value_counts(normalize=True)
                frac_etf_only = float(counts.get("ETF_ONLY", 0.0))
                frac_fut_only = float(counts.get("FUT_ONLY", 0.0))
                frac_both = float(counts.get("BOTH", 0.0))
                frac_none = float(counts.get("NONE", 0.0))

                row = {
                    "lambda_etf": lam_etf,
                    "lambda_fut": lam_fut,
                    "alpha": alpha,
                    "sharpe_annualized": summary["sharpe_annualized"],
                    "turnover_notional_per_capital": summary[
                        "turnover_notional_per_capital"
                    ],
                    "total_pnl_net": summary["total_pnl_net"],
                    "total_costs": summary["total_costs"],
                    "frac_etf_only": frac_etf_only,
                    "frac_fut_only": frac_fut_only,
                    "frac_both": frac_both,
                    "frac_none": frac_none,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


# (Optional) keep a simpler lambda/k sweep if you want
def sweep_lambda_and_k(
    base_cfg: BacktestConfig,
    lambda_etf_grid: Iterable[float],
    lambda_fut_grid: Iterable[float],
    k_l2_grid: Iterable[float],
) -> pd.DataFrame:
    """
    Older/simple sweep over (lambda_etf, lambda_fut, k_l2).
    """
    rows: List[Dict[str, Any]] = []

    for lam_etf in lambda_etf_grid:
        for lam_fut in lambda_fut_grid:
            for k_l2 in k_l2_grid:
                cfg = deepcopy(base_cfg)
                cfg.steiner.lambda_etf = lam_etf
                cfg.steiner.lambda_fut = lam_fut
                cfg.steiner.k_l2 = k_l2

                result = run_single_backtest(cfg)
                summary = result.summary

                row = {
                    "lambda_etf": lam_etf,
                    "lambda_fut": lam_fut,
                    "k_l2": k_l2,
                    "sharpe_annualized": summary["sharpe_annualized"],
                    "turnover_notional_per_capital": summary[
                        "turnover_notional_per_capital"
                    ],
                    "total_pnl_net": summary["total_pnl_net"],
                    "total_costs": summary["total_costs"],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df
