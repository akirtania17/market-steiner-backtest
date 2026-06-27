# make_synthetic_data.py

from __future__ import annotations

import numpy as np
import pandas as pd

def make_synthetic_etf_fut_data(
    n_bars: int = 10_000,
    start_price_etf: float = 100.0,
    start_price_fut: float = 15_000.0,
    etf_ann_vol: float = 0.20,
    fut_ann_vol: float = 0.25,
    corr: float = 0.9,
    minutes_per_bar: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic intraday dataset with:

      - timestamp index
      - etf_mid, fut_mid
      - etf_spread, fut_spread
      - etf_ret, fut_ret

    The ETF and FUT follow correlated geometric Brownian motion.
    Spreads drift around realistic-ish levels.
    """

    rng = np.random.default_rng(seed)

    # Time grid
    freq_str = f"{minutes_per_bar}T"
    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq=f"{minutes_per_bar}min")

    # Convert annual vol to per-bar vol (assuming 252 d * 6.5h * 60min/h ≈ 98k minutes)
    bars_per_year = 252 * int(6.5 * 60 / minutes_per_bar)
    etf_bar_vol = etf_ann_vol / np.sqrt(bars_per_year)
    fut_bar_vol = fut_ann_vol / np.sqrt(bars_per_year)

    # Correlated Gaussian shocks
    cov = np.array(
        [
            [etf_bar_vol**2, corr * etf_bar_vol * fut_bar_vol],
            [corr * etf_bar_vol * fut_bar_vol, fut_bar_vol**2],
        ]
    )
    shocks = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_bars)

    etf_rets = shocks[:, 0]
    fut_rets = shocks[:, 1]

    # Build midprice paths (GBM)
    etf_prices = start_price_etf * np.exp(np.cumsum(etf_rets))
    fut_prices = start_price_fut * np.exp(np.cumsum(fut_rets))

    # Simple spread model: mean + noise, proportional to price
    # e.g. ETF ~ 1–3 cents, FUT ~ 1–2 ticks
    etf_spread = 0.01 + 0.005 * rng.standard_normal(n_bars)  # ~1c with small noise
    etf_spread = np.clip(etf_spread, 0.005, 0.03)            # floor & cap

    fut_tick = 0.25  # e.g. NQ tick size
    fut_spread_ticks = 1.0 + 0.3 * rng.standard_normal(n_bars)
    fut_spread_ticks = np.clip(fut_spread_ticks, 0.5, 2.0)
    fut_spread = fut_spread_ticks * fut_tick

    df = pd.DataFrame(
        {
            "etf_mid": etf_prices,
            "fut_mid": fut_prices,
            "etf_spread": etf_spread,
            "fut_spread": fut_spread,
        },
        index=idx,
    )

    # Explicit returns (used by the backtester if present)
    df["etf_ret"] = df["etf_mid"].pct_change().fillna(0.0)
    df["fut_ret"] = df["fut_mid"].pct_change().fillna(0.0)

    return df


if __name__ == "__main__":
    df = make_synthetic_etf_fut_data()
    print(df.head())
    df.to_csv("synthetic_etf_fut_data.csv")
    print("Wrote synthetic_etf_fut_data.csv")
