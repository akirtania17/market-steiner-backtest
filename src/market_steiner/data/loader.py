from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import DataConfig


def load_etf_fut_csv(config: DataConfig) -> pd.DataFrame:
    """
    Load ETF–FUT 5m data from CSV with columns:
    - timestamp
    - etf_mid
    - fut_mid

    Returns a DataFrame indexed by timestamp.
    """
    df = pd.read_csv(config.csv_path)

    if config.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{config.timestamp_col}' in CSV")

    ts = pd.to_datetime(df[config.timestamp_col])
    if config.tz is not None:
        ts = ts.dt.tz_localize(config.tz) if ts.dt.tz is None else ts.dt.tz_convert(config.tz)

    df.index = ts
    df = df.sort_index()

    expected = {config.etf_col, config.fut_col}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    return df[[config.etf_col, config.fut_col]].copy()
