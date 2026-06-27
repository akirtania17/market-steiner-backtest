from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CostConfig
from ..mist.optimizer import HedgeState


@dataclass
class TradeResult:
    """
    Result of applying trades between two states, including costs.
    """
    trade_etf: float
    trade_fut: float
    cost_total: float
    cost_spread: float
    cost_commission: float
    cost_leakage: float


def compute_trade_and_cost(
    prev_state: HedgeState,
    new_state: HedgeState,
    etf_price: float,
    fut_price: float,
    cfg: CostConfig,
) -> TradeResult:
    """
    Compute trades in ETF & FUT notionals and associated costs.
    Assumes w_etf and w_fut are **dollar notionals** (not shares).

    We convert to position change in notional, then cost ~ half_spread * |Δnotional| / price,
    plus optional leakage/decay and commission.

    This is a stylized but consistent cost model; you can refine it.
    """
    w_etf_prev = prev_state.w_etf
    w_fut_prev = prev_state.w_fut
    w_etf_new = new_state.w_etf
    w_fut_new = new_state.w_fut

    delta_etf = w_etf_new - w_etf_prev
    delta_fut = w_fut_new - w_fut_prev

    # Convert notional change to approximate "shares" change
    # (divide by price). Then half-spread cost is:
    #   cost ≈ half_spread * price * |Δshares|
    #        = half_spread * |Δnotional|
    cost_spread_etf = cfg.half_spread_etf * abs(delta_etf)
    cost_spread_fut = cfg.half_spread_fut * abs(delta_fut)

    cost_spread = cost_spread_etf + cost_spread_fut

    # Commission:
    cost_commission = cfg.commission_per_notional * (abs(delta_etf) + abs(delta_fut))

    # Leakage / decay per bar, proportional to current notional (holding cost)
    leakage_etf = cfg.leakage_per_bar_etf * abs(w_etf_new)
    leakage_fut = cfg.leakage_per_bar_fut * abs(w_fut_new)
    cost_leakage = leakage_etf + leakage_fut

    total_cost = cost_spread + cost_commission + cost_leakage

    return TradeResult(
        trade_etf=float(delta_etf),
        trade_fut=float(delta_fut),
        cost_total=float(total_cost),
        cost_spread=float(cost_spread),
        cost_commission=float(cost_commission),
        cost_leakage=float(cost_leakage),
    )


def compute_step_pnl(
    prev_state: HedgeState,
    etf_ret: float,
    fut_ret: float,
) -> float:
    """
    Compute PnL for one bar, given previous-notional positions and
    realized returns on ETF and FUT.

    PnL(t) = w_prev^T * r(t)
    """
    w_etf_prev = prev_state.w_etf
    w_fut_prev = prev_state.w_fut
    pnl = w_etf_prev * etf_ret + w_fut_prev * fut_ret
    return float(pnl)
