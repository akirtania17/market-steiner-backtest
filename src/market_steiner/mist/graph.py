from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from ..config import SteinerConfig


ActivationSet = Literal["ETF_ONLY", "FUT_ONLY", "BOTH"]


@dataclass
class SteinerActivationResult:
    activation: ActivationSet
    cost_etf_only: float
    cost_fut_only: float
    cost_both: float


def steiner_activation_for_two_instruments(
    w_raw: np.ndarray,
    steiner_cfg: SteinerConfig,
    etf_fut_corr: float | None = None,
) -> SteinerActivationResult:
    """
    Simple Steiner-style activation on a 2-node graph: ETF and FUT.

    We treat:
      - T = raw target weights [w_etf_raw, w_fut_raw]
      - E = candidate that uses ETF only (FUT weight forced to 0)
      - F = candidate that uses FUT only (ETF weight forced to 0)
      - E+F = both instruments allowed

    Edges:
      - C({E})   = alpha * ||T - E|| + beta * lambda_etf
      - C({F})   = alpha * ||T - F|| + beta * lambda_fut
      - C({E,F}) = alpha * (0) + beta * (lambda_etf + lambda_fut) + optional corr term

    This is a stylized, easily tunable version of your MIST Steiner tree:
    it chooses which subset of {ETF, FUT} is even allowed to take size.

    Parameters
    ----------
    w_raw : np.ndarray
        Shape (2,) raw hedge weights [w_etf_raw, w_fut_raw].
    steiner_cfg : SteinerConfig
        Hyperparameters alpha, beta, gamma, lambda_etf, lambda_fut.
    etf_fut_corr : float, optional
        Correlation between ETF and FUT returns over some window.
        Used to optionally adjust cost of using BOTH.

    Returns
    -------
    SteinerActivationResult
        Best activation set and all three costs.
    """
    w_raw = np.asarray(w_raw, dtype=float).reshape(-1)
    if w_raw.shape[0] != 2:
        raise ValueError("w_raw must be length 2: [w_etf_raw, w_fut_raw]")

    w_etf_raw, w_fut_raw = w_raw

    # Candidate configurations
    w_E = np.array([w_etf_raw, 0.0])
    w_F = np.array([0.0, w_fut_raw])
    # For BOTH we allow both, so from the perspective of "activation cost"
    # the distance term is zero (they can match T exactly in principle).
    w_both = w_raw.copy()

    alpha = steiner_cfg.alpha
    beta = steiner_cfg.beta
    lam_etf = steiner_cfg.lambda_etf
    lam_fut = steiner_cfg.lambda_fut

    # Distances to raw target
    d_E = np.linalg.norm(w_raw - w_E)
    d_F = np.linalg.norm(w_raw - w_F)

    cost_E = alpha * d_E + beta * lam_etf
    cost_F = alpha * d_F + beta * lam_fut

    # Base BOTH cost: using both channels is flexible but not free.
    # We scale the cost with the magnitude of the desired hedge so that
    # for large w_raw, BOTH isn't automatically much cheaper than ETF_ONLY/FUT_ONLY.
    norm_raw = np.linalg.norm(w_raw)
    cost_both = alpha * norm_raw + beta * (lam_etf + lam_fut)

    # Optional correlation-based adjustment: if ETF & FUT are almost identical,
    # we reduce cost of using both. If they diverge, we can increase it.
    if etf_fut_corr is not None:
        # corr_distance = 1 - |rho|: small if highly correlated
        corr_distance = 1.0 - abs(etf_fut_corr)
        cost_both += steiner_cfg.gamma * corr_distance

    # Pick minimal cost activation set
    costs = {
        "ETF_ONLY": float(cost_E),
        "FUT_ONLY": float(cost_F),
        "BOTH": float(cost_both),
    }
    activation = min(costs, key=costs.get)

    return SteinerActivationResult(
        activation=activation,
        cost_etf_only=cost_E,
        cost_fut_only=cost_F,
        cost_both=cost_both,
    )
