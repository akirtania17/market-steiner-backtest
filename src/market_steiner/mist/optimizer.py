from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import lstsq

from ..config import SignalConfig, SteinerConfig


@dataclass
class HedgeState:
    """
    State of the hedge at a given time.
    """
    w_etf: float
    w_fut: float


def estimate_hedge_exposure(
    etf_rets: np.ndarray,
    fut_rets: np.ndarray,
    cfg: SignalConfig,
) -> float:
    """
    Estimate the hedge ratio (ETF vs FUT) via rolling regression:

        etf_ret ≈ beta * fut_ret

    Returns beta (exposure of ETF to FUT). Used to form H matrix.

    Parameters
    ----------
    etf_rets : np.ndarray
        Shape (T,) ETF returns.
    fut_rets : np.ndarray
        Shape (T,) FUT returns.
    cfg : SignalConfig
        Contains regression window and ridge_eps.

    Returns
    -------
    float
        beta_hat
    """
    assert etf_rets.shape == fut_rets.shape
    T = etf_rets.shape[0]
    if T < cfg.hedge_reg_window:
        raise ValueError("Not enough data to estimate hedge regression window")

    y = etf_rets[-cfg.hedge_reg_window :]
    x = fut_rets[-cfg.hedge_reg_window :]

    # Simple ridge regression with 1 regressor (plus intercept).
    X = np.column_stack([x, np.ones_like(x)])
    ridge = cfg.ridge_eps

    # (X^T X + ridge I)^{-1} X^T y; but we can fudge with lstsq by augmenting
    # or do explicit formula. Simpler: use normal equations with ridge.
    XtX = X.T @ X
    XtX[0, 0] += ridge  # ridge only on beta coeff
    Xty = X.T @ y
    params = np.linalg.solve(XtX, Xty)
    beta_hat = float(params[0])
    return beta_hat


def compute_raw_hedge(
    target_ret: float,
    beta_fut: float,
) -> np.ndarray:
    """
    Compute raw continuous hedge weights w_raw for [ETF, FUT].

    We consider n=1 dimension (one factor/return we want to match):
        r_target ≈ h_etf * w_etf + h_fut * w_fut

    Take:
        h_etf = 1
        h_fut = beta_fut

    and solve:
        min_w (r_target - [1, beta_fut] @ w)^2

    Any w on the line { w_etf + beta_fut * w_fut = r_target } is equivalent
    in terms of hedge error. We pick the minimum-norm solution:

        minimize ||w||^2 subject to h @ w = r_target.

    This is a simple projection:
        w_raw = r_target * h / ||h||^2
    """
    h = np.array([1.0, beta_fut], dtype=float)
    denom = float(h @ h)
    if denom <= 1e-12:
        # If beta is effectively zero, we can't rely on FUT. Just hedge with ETF.
        return np.array([target_ret, 0.0], dtype=float)

    # Base minimum-norm solution in "return units"
    w_raw = float(target_ret) * h / denom

    # ---- NEW: amplify ETF leg so Steiner has a real choice ----
    ETF_AMPLIFICATION = 1.2182  # tune this (20, 50, 100, etc.)
    w_raw[0] *= ETF_AMPLIFICATION
    # -----------------------------------------------------------

    return w_raw

def optimize_final_hedge_multi(
    w_raw: np.ndarray,
    prev_w: np.ndarray,
    steiner_cfg: SteinerConfig,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    N-dimensional version of optimize_final_hedge.

    Objective (per-bar):

        J(w) = ||w - w_raw||^2 + k_l2 * ||w - w_prev||^2

    subject to:
        w_i = 0 for any instrument i with active_mask[i] == False.

    Parameters
    ----------
    w_raw : np.ndarray
        Shape (M,). Raw desired hedge weights (notionals) for M instruments.
    prev_w : np.ndarray
        Shape (M,). Previous hedge weights.
    steiner_cfg : SteinerConfig
        Contains k_l2.
    active_mask : np.ndarray
        Boolean mask of shape (M,), True where instrument is allowed to trade.

    Returns
    -------
    w_new : np.ndarray
        Shape (M,). New hedge weights after smoothing and activation.
    """
    w_raw = np.asarray(w_raw, dtype=float).reshape(-1)
    prev_w = np.asarray(prev_w, dtype=float).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    if w_raw.shape[0] != prev_w.shape[0] or w_raw.shape[0] != active_mask.shape[0]:
        raise ValueError("w_raw, prev_w, and active_mask must have same length")

    k_l2 = steiner_cfg.k_l2
    if k_l2 <= 0.0:
        w_unconstrained = w_raw
    else:
        w_unconstrained = (w_raw + k_l2 * prev_w) / (1.0 + k_l2)

    # Apply activation: zero out inactivated instruments
    w_new = np.where(active_mask, w_unconstrained, 0.0)
    return w_new

def optimize_final_hedge(
    w_raw: np.ndarray,
    prev_state: HedgeState,
    steiner_cfg: SteinerConfig,
    activation: str,
) -> HedgeState:
    """
    Given the raw hedge weights w_raw (shape (2,)) and the previous hedge state,
    compute the final hedge weights (w_etf, w_fut) after:

      - Restricting to the Steiner activation set A ∈ {ETF_ONLY, FUT_ONLY, BOTH}
      - Applying a quadratic smoothness penalty on Δw

    Objective (for this 1-step update):

        J(w) = ||w - w_raw||^2 + k_l2 * ||w - w_prev||^2

    subject to:
        - If ETF_ONLY: w_fut = 0
        - If FUT_ONLY: w_etf = 0
        - If BOTH: both are free

    This is a small convex problem with closed-form solutions for all three cases.
    """
    w_raw = np.asarray(w_raw, dtype=float).reshape(-1)
    if w_raw.shape[0] != 2:
        raise ValueError("w_raw must be length 2: [w_etf_raw, w_fut_raw]")

    w_prev = np.array([prev_state.w_etf, prev_state.w_fut], dtype=float)
    k_l2 = steiner_cfg.k_l2

    # If no trading cost penalty, just project w_raw onto activation set.
    if k_l2 <= 0.0:
        if activation == "ETF_ONLY":
            return HedgeState(w_etf=float(w_raw[0]), w_fut=0.0)
        elif activation == "FUT_ONLY":
            return HedgeState(w_etf=0.0, w_fut=float(w_raw[1]))
        else:
            return HedgeState(w_etf=float(w_raw[0]), w_fut=float(w_raw[1]))

    # In general, solve:
    #   min_w ||w - w_raw||^2 + k ||w - w_prev||^2
    # = (1 + k) ||w - (w_raw + k w_prev)/(1+k)||^2 + const
    # So unconstrained solution is:
    #   w_star = (w_raw + k w_prev)/(1 + k)
    w_unconstrained = (w_raw + k_l2 * w_prev) / (1.0 + k_l2)

    if activation == "ETF_ONLY":
        w_etf = w_unconstrained[0]
        return HedgeState(w_etf=float(w_etf), w_fut=0.0)
    elif activation == "FUT_ONLY":
        w_fut = w_unconstrained[1]
        return HedgeState(w_etf=0.0, w_fut=float(w_fut))
    elif activation == "BOTH":
        return HedgeState(w_etf=float(w_unconstrained[0]),
                          w_fut=float(w_unconstrained[1]))
    else:
        raise ValueError(f"Unknown activation mode: {activation}")
