# src/market_steiner/mist/multinode.py
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import SteinerConfig


def steiner_select_subset_multi(
    w_raw: np.ndarray,
    lambdas: np.ndarray,
    steiner_cfg: SteinerConfig,
    corr_matrix: Optional[np.ndarray] = None,
    min_notional_threshold: float = 0.0,
) -> np.ndarray:
    """
    Multi-node Steiner-style activation for an ETF basket.

    Parameters
    ----------
    w_raw : np.ndarray
        Shape (M,). Raw desired hedge notionals for M instruments.
    lambdas : np.ndarray
        Shape (M,). Leakage λ_i for each instrument.
    steiner_cfg : SteinerConfig
        Uses alpha, beta, gamma.
    corr_matrix : Optional[np.ndarray]
        Shape (M, M). Correlation matrix between instrument returns.
        Used only if gamma > 0.
    min_notional_threshold : float
        If included notionals ||w_raw[subset]|| < threshold, skip that subset.

    Returns
    -------
    active_mask : np.ndarray
        Boolean mask of shape (M,) indicating which instruments are in
        the chosen Steiner subset. If no subset passes threshold, all False.
    """
    w_raw = np.asarray(w_raw, dtype=float).reshape(-1)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    M = w_raw.shape[0]
    if lambdas.shape[0] != M:
        raise ValueError("lambdas must have same length as w_raw")

    alpha = steiner_cfg.alpha
    beta = steiner_cfg.beta
    gamma = steiner_cfg.gamma

    best_cost = np.inf
    best_mask = np.zeros(M, dtype=bool)

    # Enumerate all non-empty subsets of {0, ..., M-1}
    # For small M (<= 6), this is fine: 2^M - 1 possibilities.
    for mask_int in range(1, 1 << M):
        indices = [i for i in range(M) if (mask_int & (1 << i))]

        # Included notional norm: how big this subset actually is
        w_included = w_raw[indices]
        w_included_norm = float(np.linalg.norm(w_included))
        if w_included_norm < min_notional_threshold:
            # Ignore tiny, irrelevant subsets
            continue

        # EXCLUDED notional: how much of the raw hedge we are throwing away
        w_excluded = w_raw.copy()
        w_excluded[indices] = 0.0

        # Distance cost: penalize discarding large chunks of the raw hedge
        cost_dist = alpha * float(np.linalg.norm(w_excluded))

        # Leakage cost: sum λ_i for active instruments (you can weight by |w_i| if desired)
        cost_leak = beta * float(lambdas[indices].sum())

        # Correlation (Steiner tree structure) cost
        cost_corr = 0.0
        if corr_matrix is not None and gamma > 0.0 and len(indices) > 1:
            # penalize subsets that mix low-correlation instruments,
            # weighted by average absolute notional of each pair
            for a_pos in range(len(indices)):
                for b_pos in range(a_pos + 1, len(indices)):
                    i = indices[a_pos]
                    j = indices[b_pos]
                    rho = float(corr_matrix[i, j])
                    corr_dist = 1.0 - rho * rho  # like (1 - ρ^2)
                    avg_weight = 0.5 * (abs(w_raw[i]) + abs(w_raw[j]))
                    cost_corr += gamma * corr_dist * avg_weight

        total_cost = cost_dist + cost_leak + cost_corr

        if total_cost < best_cost:
            best_cost = total_cost
            best_mask[:] = False
            best_mask[indices] = True

    # If we never found a subset above threshold, return NONE (no activation)
    if not np.any(best_mask):
        return np.zeros(M, dtype=bool)

    return best_mask
