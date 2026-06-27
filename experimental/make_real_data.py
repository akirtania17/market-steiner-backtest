# make_real_data.py
#
# Download real intraday data for an ETF + proxy (e.g. XLK vs QQQ)
# and save it as:
#   index: timestamp
#   columns: etf_mid, fut_mid

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf


def download_intraday(
    ticker: str,
    period: str = "60d",   # rolling window, avoids old-date restriction
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Download intraday bars via yfinance.

    Yahoo rules of thumb:
      - 1m: last ~7 days
      - 2m / 5m: last ~60 days
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} (period={period}, interval={interval})."
        )

    # Use Close as proxy for mid
    df = df[["Close"]].rename(columns={"Close": "mid"})
    df.index.name = "timestamp"
    return df


def make_real_etf_fut_dataset(
    etf_ticker: str = "XLK",
    fut_ticker: str = "QQQ",  # treat QQQ as futures-like proxy
    period: str = "60d",
    interval: str = "5m",
    out_path: str = "data/real_etf_fut_5m.csv",
) -> pd.DataFrame:
    print(f"Downloading {etf_ticker} (period={period}, interval={interval})...")
    df_etf = download_intraday(etf_ticker, period=period, interval=interval)

    print(f"Downloading {fut_ticker} (period={period}, interval={interval})...")
    df_fut = download_intraday(fut_ticker, period=period, interval=interval)

    # Align timestamps and FORCE column names
    df = pd.concat(
        [df_etf["mid"], df_fut["mid"]],
        axis=1,
        join="inner",
    )
    df.columns = ["etf_mid", "fut_mid"]  # <-- guarantees these names exist

    # Keep US cash session only (optional, but nice)
    df = df.between_time("09:30", "16:00")

    # Drop any missing values
    df = df.dropna()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=True)
    print(f"Saved real data to {out_path.resolve()}  (rows={len(df)})")

    return df


if __name__ == "__main__":
    make_real_etf_fut_dataset()
