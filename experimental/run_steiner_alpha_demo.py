# run_steiner_alpha_demo.py

import pandas as pd
from make_synthetic_data import make_synthetic_etf_fut_data
from steiner_alpha_backtest import AlphaConfig, SteinerAlphaBacktester

def main():
    # 1) Make synthetic data
    df = make_synthetic_etf_fut_data(
        n_bars=20_000,
        start_price_etf=100.0,
        start_price_fut=15_000.0,
        etf_ann_vol=0.18,
        fut_ann_vol=0.22,
        corr=0.92,
        minutes_per_bar=1,
        seed=123,
    )

    # 2) Config that matches your Steiner universe
    cfg = AlphaConfig(
        n_stocks=25,
        lambda_etf_leakage=0.1,
        lambda_fut_leakage=0.08,
        sector_reduction=-0.12,
        etf_cap_fraction=0.25,
        fut_cap_fraction=0.25,
        notional_unit=1_000_000.0,
        signal_k=1.0,
        vol_lookback=60,
        max_gross_leverage=3.0,
        trade_cost_bps=0.2,
    )

    # 3) Run backtest
    bt = SteinerAlphaBacktester(df, cfg)
    result = bt.run(initial_capital=1_000_000.0)

    print("=== Backtest summary ===")
    for k, v in result.summary.items():
        print(f"{k}: {v:.4f}")

    print("\nSample trades:")
    print(result.trades.head())

    # Optional: plot (if you're running in a notebook or python shell)
    try:
        import matplotlib.pyplot as plt

        result.equity_curve.plot(title="Steiner ETF–FUT Alpha Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.show()
    except Exception as e:
        print("Plotting failed (no matplotlib or headless env):", e)


if __name__ == "__main__":
    main()
