from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from ..config import SignalConfig, SteinerConfig, CostConfig, BacktestConfig
from ..mist.optimizer import HedgeState, estimate_hedge_exposure, compute_raw_hedge, optimize_final_hedge
from ..mist.graph import steiner_activation_for_two_instruments
from .costs import compute_trade_and_cost, compute_step_pnl, TradeResult


@dataclass
class HedgeStepLog:
    timestamp: pd.Timestamp
    w_etf: float
    w_fut: float
    trade_etf: float
    trade_fut: float
    pnl_gross: float
    cost_total: float
    equity: float
    activation: str
    beta_fut: float
    raw_w_etf: float
    raw_w_fut: float
    cost_act_etf_only: float
    cost_act_fut_only: float
    cost_act_both: float


class SteinerHedgeStrategy:
    """
    ETF–FUT Steiner hedge strategy:

    - Uses rolling regression to estimate ETF vs FUT beta.
    - Computes a raw hedge weight vector w_raw = [w_etf_raw, w_fut_raw].
    - Uses a Steiner-like activation rule to choose which instruments can trade.
    - Applies a smooth quadratic adjustment toward w_raw, constrained by activation set.
    - Trades on the next bar and accrues PnL and costs.

    All notionals are in dollars relative to cfg.costs.initial_capital.
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.state = HedgeState(w_etf=0.0, w_fut=0.0)
        self.equity = cfg.costs.initial_capital
        self.logs: List[HedgeStepLog] = []
        self._initialized = False

    def reset(self):
        self.state = HedgeState(w_etf=0.0, w_fut=0.0)
        self.equity = self.cfg.costs.initial_capital
        self.logs.clear()
        self._initialized = False

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entrypoint: run strategy on a DataFrame with columns:
        - etf_mid
        - fut_mid
        - etf_mid_ret
        - fut_mid_ret

        Indexed by timestamp.
        """
        self.reset()
        cfg_sig = self.cfg.signal
        n = len(df)
        if n <= cfg_sig.hedge_reg_window + 1:
            raise ValueError("Not enough data for rolling hedge regression.")

        # We'll iterate from index `start_idx` onward, using prior window for regression.
        start_idx = cfg_sig.hedge_reg_window

        etf_prices = df["etf_mid"].to_numpy()
        fut_prices = df["fut_mid"].to_numpy()
        etf_rets = df["etf_mid_ret"].to_numpy()
        fut_rets = df["fut_mid_ret"].to_numpy()

        timestamps = df.index.to_list()

        for t in range(start_idx + 1, n):
            ts = timestamps[t]

            # 1. Estimate hedge exposure beta over window [t - window, t-1]
            window_slice = slice(t - cfg_sig.hedge_reg_window, t)
            beta_fut = estimate_hedge_exposure(
                etf_rets[window_slice],
                fut_rets[window_slice],
                cfg_sig,
            )

            # 2. Define target return to hedge.
            # For simplicity, we take the ETF return at t as the thing we "want to replicate/hedge".
            target_ret = etf_rets[t]

            # 3. Compute raw continuous hedge weights (in "return units")
            w_raw = compute_raw_hedge(target_ret=target_ret, beta_fut=beta_fut)

            # Scale raw hedge from "return units" to dollar notionals.
            # Here we interpret w_raw as fraction of capital.
            capital = self.cfg.costs.initial_capital
            w_raw = w_raw * capital

            # Optional no-trade threshold on tiny hedges
            if np.linalg.norm(w_raw) < self.cfg.steiner.min_notional_threshold:
                # Hold previous state, only mark PnL from existing positions
                step_pnl = compute_step_pnl(
                    prev_state=self.state,
                    etf_ret=etf_rets[t],
                    fut_ret=fut_rets[t],
                )
                # No trade cost if no trades
                self.equity += step_pnl
                self.logs.append(
                    HedgeStepLog(
                        timestamp=ts,
                        w_etf=self.state.w_etf,
                        w_fut=self.state.w_fut,
                        trade_etf=0.0,
                        trade_fut=0.0,
                        pnl_gross=step_pnl,
                        cost_total=0.0,
                        equity=self.equity,
                        activation="NONE",
                        beta_fut=beta_fut,
                        raw_w_etf=w_raw[0],
                        raw_w_fut=w_raw[1],
                        cost_act_etf_only=0.0,
                        cost_act_fut_only=0.0,
                        cost_act_both=0.0,
                    )
                )
                continue

            # 4. Compute correlation between ETF & FUT in the same window
            corr = np.corrcoef(
                etf_rets[window_slice],
                fut_rets[window_slice],
            )[0, 1]

            # 5. Steiner activation
            act_res = steiner_activation_for_two_instruments(
                w_raw=w_raw,
                steiner_cfg=self.cfg.steiner,
                etf_fut_corr=corr,
            )

            # 6. Final hedge optimization under activation
            new_state = optimize_final_hedge(
                w_raw=w_raw,
                prev_state=self.state,
                steiner_cfg=self.cfg.steiner,
                activation=act_res.activation,
            )

            # 7. Compute trade & cost moving from prev_state to new_state
            trade_res: TradeResult = compute_trade_and_cost(
                prev_state=self.state,
                new_state=new_state,
                etf_price=etf_prices[t],
                fut_price=fut_prices[t],
                cfg=self.cfg.costs,
            )

            # 8. Compute PnL for this bar based on previous positions
            step_pnl = compute_step_pnl(
                prev_state=self.state,
                etf_ret=etf_rets[t],
                fut_ret=fut_rets[t],
            )

            # 9. Update equity and state
            self.equity += step_pnl - trade_res.cost_total
            self.state = new_state

            # 10. Log
            self.logs.append(
                HedgeStepLog(
                    timestamp=ts,
                    w_etf=new_state.w_etf,
                    w_fut=new_state.w_fut,
                    trade_etf=trade_res.trade_etf,
                    trade_fut=trade_res.trade_fut,
                    pnl_gross=step_pnl,
                    cost_total=trade_res.cost_total,
                    equity=self.equity,
                    activation=act_res.activation,
                    beta_fut=beta_fut,
                    raw_w_etf=w_raw[0],
                    raw_w_fut=w_raw[1],
                    cost_act_etf_only=act_res.cost_etf_only,
                    cost_act_fut_only=act_res.cost_fut_only,
                    cost_act_both=act_res.cost_both,
                )
            )

        # Convert logs to DataFrame
        log_df = pd.DataFrame([log.__dict__ for log in self.logs])
        log_df = log_df.set_index("timestamp")
        return log_df
