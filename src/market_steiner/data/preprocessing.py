from __future__ import annotations

import numpy as np
import pandas as pd


def align_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop rows where either ETF or FUT is missing.
    - Optionally, you could forward-fill with a max-gap logic. For now, we keep it strict.
    """
    cleaned = df.dropna(subset=df.columns)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    return cleaned


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns for both ETF and FUT:
    r(t) = log(price(t)) - log(price(t-1))
    """
    log_prices = np.log(df)
    rets = log_prices.diff()
    rets.columns = [f"{c}_ret" for c in df.columns]
    return rets


def merge_prices_and_returns(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and return frames into a single frame for convenience.
    """
    df = prices.join(returns, how="inner")
    df = df.dropna()
    return df
