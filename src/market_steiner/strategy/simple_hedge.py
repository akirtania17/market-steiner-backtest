from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..mist.optimizer import (
    HedgeState,
    estimate_hedge_exposure,
    compute_raw_hedge,
    optimize_final_hedge,
)
from .costs import compute_trade_and_cost, compute_step_pnl, TradeResult


@dataclass
class SimpleHedgeStepLog:
    timestamp: pd.Timestamp
    w_etf: float
    w_fut: float
    trade_etf: float
    trade_fut: float
    pnl_gross: float
    cost_total: float
    equity: float
    beta_fut: float
    raw_w_etf: float
    raw_w_fut: float


class SimpleHedgeStrategy:
    """
    Baseline hedge: use both ETF and FUT with no Steiner activation.

    - Same rolling regression for beta_fut.
    - Same raw hedge computation.
    - Same k_l2 smoothing.
    - No activation choice: always activation="BOTH" in optimize_final_hedge.
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.state = HedgeState(w_etf=0.0, w_fut=0.0)
        self.equity = cfg.costs.initial_capital
        self.logs: List[SimpleHedgeStepLog] = []
        self._initialized = False

    def reset(self):
        self.state = HedgeState(w_etf=0.0, w_fut=0.0)
        self.equity = self.cfg.costs.initial_capital
        self.logs.clear()
        self._initialized = False

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run baseline hedge on DataFrame with columns:
        - etf_mid, fut_mid, etf_mid_ret, fut_mid_ret
        """
        self.reset()
        cfg_sig = self.cfg.signal
        n = len(df)
        if n <= cfg_sig.hedge_reg_window + 1:
            raise ValueError("Not enough data for rolling hedge regression.")

        start_idx = cfg_sig.hedge_reg_window

        etf_prices = df["etf_mid"].to_numpy()
        fut_prices = df["fut_mid"].to_numpy()
        etf_rets = df["etf_mid_ret"].to_numpy()
        fut_rets = df["fut_mid_ret"].to_numpy()

        timestamps = df.index.to_list()

        for t in range(start_idx + 1, n):
            ts = timestamps[t]

            # 1. Rolling regression for beta_fut
            window_slice = slice(t - cfg_sig.hedge_reg_window, t)
            beta_fut = estimate_hedge_exposure(
                etf_rets[window_slice],
                fut_rets[window_slice],
                cfg_sig,
            )

            # 2. Target return to hedge (same as Steiner strategy)
            target_ret = etf_rets[t]

            # 3. Raw hedge in return units
            w_raw = compute_raw_hedge(target_ret=target_ret, beta_fut=beta_fut)

            # 4. Scale to dollar notionals (same as Steiner strategy)
            capital = self.cfg.costs.initial_capital
            w_raw = w_raw * capital

            # 5. Apply no-trade threshold (reuse same minimum threshold)
            if np.linalg.norm(w_raw) < self.cfg.steiner.min_notional_threshold:
                step_pnl = compute_step_pnl(
                    prev_state=self.state,
                    etf_ret=etf_rets[t],
                    fut_ret=fut_rets[t],
                )
                self.equity += step_pnl
                self.logs.append(
                    SimpleHedgeStepLog(
                        timestamp=ts,
                        w_etf=self.state.w_etf,
                        w_fut=self.state.w_fut,
                        trade_etf=0.0,
                        trade_fut=0.0,
                        pnl_gross=step_pnl,
                        cost_total=0.0,
                        equity=self.equity,
                        beta_fut=beta_fut,
                        raw_w_etf=w_raw[0],
                        raw_w_fut=w_raw[1],
                    )
                )
                continue

            # 6. Final hedge optimization with activation="BOTH" (no Steiner)
            new_state = optimize_final_hedge(
                w_raw=w_raw,
                prev_state=self.state,
                steiner_cfg=self.cfg.steiner,
                activation="BOTH",
            )

            # 7. Trades & costs
            trade_res: TradeResult = compute_trade_and_cost(
                prev_state=self.state,
                new_state=new_state,
                etf_price=etf_prices[t],
                fut_price=fut_prices[t],
                cfg=self.cfg.costs,
            )

            # 8. PnL from previous positions
            step_pnl = compute_step_pnl(
                prev_state=self.state,
                etf_ret=etf_rets[t],
                fut_ret=fut_rets[t],
            )

            # 9. Update
            self.equity += step_pnl - trade_res.cost_total
            self.state = new_state

            # 10. Log
            self.logs.append(
                SimpleHedgeStepLog(
                    timestamp=ts,
                    w_etf=new_state.w_etf,
                    w_fut=new_state.w_fut,
                    trade_etf=trade_res.trade_etf,
                    trade_fut=trade_res.trade_fut,
                    pnl_gross=step_pnl,
                    cost_total=trade_res.cost_total,
                    equity=self.equity,
                    beta_fut=beta_fut,
                    raw_w_etf=w_raw[0],
                    raw_w_fut=w_raw[1],
                )
            )

        log_df = pd.DataFrame([log.__dict__ for log in self.logs])
        log_df = log_df.set_index("timestamp")
        return log_df
