# src/market_steiner/strategy/multi_basket.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig, BasketConfig, BasketInstrumentConfig
from ..mist.optimizer import optimize_final_hedge_multi
from ..mist.multinode import steiner_select_subset_multi


@dataclass
class BasketHedgeState:
    w: np.ndarray  # shape (M,)


@dataclass
class BasketStepLog:
    timestamp: pd.Timestamp
    w: Dict[str, float]
    trades: Dict[str, float]
    pnl_gross: float
    cost_total: float
    equity: float
    active_mask: Dict[str, bool]


class MultiEtfSteinerStrategy:
    """
    Multi-instrument Steiner-activated hedge strategy.

    - Instruments: arbitrary mix of ETFs, futures, etc.
    - Raw hedge: solved via ridge-regularized least squares each bar:

          y_window ≈ X_window @ w

      where y_window are target returns and X_window are instrument returns.

    - Steiner: chooses the cheapest subset of instruments to activate
      given w_raw, lambdas, alpha/beta/gamma, and correlation matrix.
    """

    def __init__(self, cfg: BacktestConfig, basket_cfg: BasketConfig):
        self.cfg = cfg
        self.basket_cfg = basket_cfg
        self.state: Optional[BasketHedgeState] = None
        self.equity: float = cfg.costs.initial_capital
        self.logs: List[BasketStepLog] = []

    def reset(self):
        M = len(self.basket_cfg.instruments)
        self.state = BasketHedgeState(w=np.zeros(M, dtype=float))
        self.equity = self.cfg.costs.initial_capital
        self.logs.clear()

    # ---------- raw hedge: LS "QP-like" solver ----------

    def _compute_raw_hedge_vector_ls(
        self,
        t: int,
        target_rets: np.ndarray,  # shape (T,)
        ret_mat: np.ndarray,      # shape (T, M)
    ) -> np.ndarray:
        """
        Solve a ridge-regularized LS hedge:

            y ≈ X @ w

        where:
            - y: past window of target returns (length W)
            - X: past window of instrument returns (W x M)
            - w: hedge weights in "return units"

        We then scale by capital to get notionals.
        """
        sig_cfg = self.cfg.signal
        W = sig_cfg.hedge_reg_window
        ridge_eps = sig_cfg.ridge_eps

        if t < W:
            # Not enough history yet
            return np.zeros(ret_mat.shape[1], dtype=float)

        # Window [t-W+1, ..., t]
        start = t - W + 1
        stop = t + 1

        y = target_rets[start:stop]                # shape (W,)
        X = ret_mat[start:stop, :]                 # shape (W, M)

        # XTX + ridge * I
        XTX = X.T @ X
        M = XTX.shape[0]
        XTX += ridge_eps * np.eye(M)

        XTy = X.T @ y

        try:
            w_returns = np.linalg.solve(XTX, XTy)  # shape (M,)
        except np.linalg.LinAlgError:
            # Fallback: zero hedge if matrix ill-conditioned
            return np.zeros(ret_mat.shape[1], dtype=float)

        # Scale into notional space
        capital = self.cfg.costs.initial_capital
        w_raw = capital * w_returns
        return w_raw

    # ---------- main loop ----------

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the multi-ETF Steiner strategy over a DataFrame.

        Requirements on df columns:
          - basket_cfg.target_ret_col
          - for each instrument in basket_cfg.instruments:
                - instr.ret_col (returns)
                - instr.price_col (for cost model if you extend it)
        """
        self.reset()

        instrs: List[BasketInstrumentConfig] = self.basket_cfg.instruments
        M = len(instrs)
        lambdas = np.array([instr.lambda_leak for instr in instrs], dtype=float)

        # Build instrument return matrix: shape (T, M)
        ret_mat = np.column_stack([df[instr.ret_col].to_numpy() for instr in instrs])
        target_rets = df[self.basket_cfg.target_ret_col].to_numpy()
        timestamps = df.index.to_list()

        # Precompute correlation matrix over entire sample (you can make this rolling later)
        corr_matrix = np.corrcoef(ret_mat.T)

        for t in range(len(df)):
            ts = timestamps[t]

            prev_w = self.state.w

            # 1) Raw hedge in instrument space (notionals)
            w_raw = self._compute_raw_hedge_vector_ls(
                t=t,
                target_rets=target_rets,
                ret_mat=ret_mat,
            )

            # 2) Multi-node Steiner selection (subset activation)
            active_mask = steiner_select_subset_multi(
                w_raw=w_raw,
                lambdas=lambdas,
                steiner_cfg=self.cfg.steiner,
                corr_matrix=corr_matrix,
                min_notional_threshold=self.cfg.steiner.min_notional_threshold,
            )

            # 3) Optimize final hedge under activation + L2 smoothness
            w_new = optimize_final_hedge_multi(
                w_raw=w_raw,
                prev_w=prev_w,
                steiner_cfg=self.cfg.steiner,
                active_mask=active_mask,
            )

            trades = w_new - prev_w

            # 4) PnL model (placeholder):
            #    Here we use the instrument returns directly:
            #       pnl_gross = sum( prev_w[i] * ret_mat[t, i] )
            #    You can refine this to include target vs hedge basis if desired.
            pnl_gross = float(prev_w @ ret_mat[t, :])

            # TODO: implement multi-instrument transaction/leakage costs.
            # For now set to 0 to focus on hedge logic.
            cost_total = 0.0

            self.equity += pnl_gross - cost_total
            self.state = BasketHedgeState(w=w_new)

            self.logs.append(
                BasketStepLog(
                    timestamp=ts,
                    w={instrs[i].name: float(w_new[i]) for i in range(M)},
                    trades={instrs[i].name: float(trades[i]) for i in range(M)},
                    pnl_gross=pnl_gross,
                    cost_total=cost_total,
                    equity=self.equity,
                    active_mask={instrs[i].name: bool(active_mask[i]) for i in range(M)},
                )
            )

        log_df = pd.DataFrame([log.__dict__ for log in self.logs])
        log_df = log_df.set_index("timestamp")
        return log_df
