import os
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "multi_etf_fut_5m.csv")

TICKERS = {
    "qqq": "QQQ",
    "xlk": "XLK",
    "nq": "NQ=F",  # Nasdaq futures continuous
}

PERIOD = "60d"
INTERVAL = "5m"


def download_one(label: str, ticker: str) -> pd.DataFrame:
    print(f"Downloading {label} ({ticker})...")
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL)
    # Use Adj Close or Close as mid
    df = df[["Close"]].rename(columns={"Close": f"{label}_mid"})
    return df


def main():
    dfs = []
    for label, ticker in TICKERS.items():
        df = download_one(label, ticker)
        dfs.append(df)

    # Align all on datetime index (inner join)
    df_all = pd.concat(dfs, axis=1, join="inner")

    # Compute 5m returns for each instrument
    for label in TICKERS.keys():
        mid_col = f"{label}_mid"
        ret_col = f"{label}_ret"
        df_all[ret_col] = df_all[mid_col].pct_change().fillna(0.0)

    # Move index to a real column called "timestamp"
    df_all = df_all.reset_index().rename(columns={"Datetime": "timestamp"})

    # Save clean CSV
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_csv(OUT_PATH, index=False)

    print(f"\nSaved clean multi-instrument file to:\n  {OUT_PATH}")
    print(df_all.head())


if __name__ == "__main__":
    main()
