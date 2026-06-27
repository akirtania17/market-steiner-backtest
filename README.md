# market_steiner_backtest

An ETF-versus-futures hedge backtester that uses a Steiner-tree-style activation rule to decide which instruments to trade.

## Overview

This project backtests a hedging strategy on intraday (5-minute) ETF and equity-index-future data. Each bar it estimates the ETF-vs-future hedge ratio with a rolling ridge regression, builds a raw continuous hedge target, and then runs a small Steiner-tree-style activation decision to choose which instruments are even allowed to take size: the ETF only, the future only, or both. The activation choice minimizes a cost that trades off distance-to-target against per-instrument leakage cost and a correlation penalty. The strategy then solves a small convex update toward the target on the activated instruments, trades on the next bar, and accrues PnL net of a stylized cost model.

A simple baseline (always trade both instruments, no activation) is included so the activation logic can be compared against a plain hedge on the same data and cost model.

This is a research backtest on the small sample of data included in the repo. It is not a live trading system and the numbers below are in-sample backtest figures, not realized returns.

## How it works

The core loop lives in `src/market_steiner/backtest/engine.py` and `src/market_steiner/strategy/steiner_hedge.py`.

### 1. Hedge-ratio estimation

For each bar `t`, the strategy takes the prior window of returns (default 200 bars, `SignalConfig.hedge_reg_window`) and regresses ETF returns on future returns:

```
etf_ret ~= beta * fut_ret + intercept
```

The regression is a ridge-regularized normal-equations solve (`estimate_hedge_exposure` in `src/market_steiner/mist/optimizer.py`), with the ridge term applied only to the slope coefficient (`SignalConfig.ridge_eps`). The result is `beta_fut`, the exposure of the ETF to the future.

The raw hedge target (`compute_raw_hedge`) treats the current ETF return as the thing to replicate and forms a minimum-norm solution along the line `w_etf + beta_fut * w_fut = target_ret`, then scales by capital to get dollar notionals. The ETF leg is amplified by a fixed factor so the activation decision has a real tradeoff to make rather than always preferring the cheaper leg.

### 2. Steiner activation cost model

The activation decision is in `src/market_steiner/mist/graph.py` (`steiner_activation_for_two_instruments`). Given the raw target `w_raw = [w_etf, w_fut]`, it scores three candidate activation sets:

```
C(ETF_ONLY) = alpha * ||w_raw - [w_etf, 0]||   + beta * lambda_etf
C(FUT_ONLY) = alpha * ||w_raw - [0, w_fut]||   + beta * lambda_fut
C(BOTH)     = alpha * ||w_raw||                 + beta * (lambda_etf + lambda_fut) + gamma * (1 - |corr|)
```

`alpha` weights the distance between the raw target and the candidate (how much of the desired hedge is thrown away), `beta` weights the per-instrument leakage penalties `lambda_etf` / `lambda_fut`, and `gamma` adds a correlation penalty to the BOTH option that shrinks when the ETF and future are highly correlated. The cheapest of the three is selected. This is the two-node version of the more general subset selection in `src/market_steiner/mist/multinode.py`, which enumerates all non-empty subsets of an instrument basket and scores distance + leakage + a pairwise `(1 - rho^2)` correlation cost.

### 3. Final hedge update

Given the activation set, `optimize_final_hedge` solves a one-step convex update:

```
min_w  ||w - w_raw||^2 + k_l2 * ||w - w_prev||^2
```

subject to zeroing out any instrument not in the activation set. The `k_l2` term is a smoothness penalty on the change in position that damps churn. The closed-form solution is `(w_raw + k_l2 * w_prev) / (1 + k_l2)`, projected onto the active instruments.

### 4. Backtest loop, costs, PnL

For each bar the engine:

- estimates `beta_fut`, builds `w_raw`, and applies a no-trade band (`min_notional_threshold`) so tiny hedges hold the prior position
- computes the activation set and the final target weights
- computes trades as the change in notional and a stylized cost (half-spread on traded notional, optional commission, optional per-bar leakage) in `src/market_steiner/strategy/costs.py`
- marks PnL on the previous bar's positions: `pnl = w_etf_prev * etf_ret + w_fut_prev * fut_ret`
- updates equity by `pnl - cost`

Metrics (`src/market_steiner/backtest/metrics.py`) are annualized Sharpe on net per-bar PnL, max drawdown, and notional turnover per unit of capital. Sharpe is annualized with `bars_per_year = 252 * 78` (5-minute bars).

## Tech stack

- Python 3
- numpy and pandas for the core engine
- yfinance for downloading the sample data (data-build scripts only)
- matplotlib for optional plotting in the experimental scripts

## Setup

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` is currently empty. The runtime dependencies are `numpy` and `pandas`; the data-build scripts additionally need `yfinance`, and the experimental scripts use `matplotlib`, `networkx`, and `tqdm`.

The scripts add `src/` to `sys.path` themselves, so no install step beyond the dependencies is required. Run them from the repo root.

## Usage

All entry points are in `scripts/`.

Single pair backtest with the default config:

```bash
python scripts/run_single_pair.py
```

Compare the Steiner strategy against the always-trade-both baseline:

```bash
python scripts/run_compare_baseline.py
```

Phase sweep over `(lambda_etf, lambda_fut, alpha)`, writes `phase_sweep_results.csv`:

```bash
python scripts/run_phase_sweep.py
```

Older sweep over `(lambda_etf, lambda_fut, k_l2)`:

```bash
python scripts/run_param_sweep.py
```

Multi-instrument basket backtest (hedge XLK using QQQ + NQ), reads `data/multi_etf_fut_5m.csv`:

```bash
python scripts/run_multi_basket.py
```

Rebuild the multi-instrument dataset from yfinance (needs network access):

```bash
python scripts/build_multi_csv.py
```

## Project structure

```
market_steiner_backtest/
  README.md
  requirements.txt
  phase_sweep_results.csv        # saved phase-sweep output (portfolio evidence)
  data/
    raw/real_etf_fut_5m.csv      # single-pair ETF/FUT 5m sample
    multi_etf_fut_5m.csv         # multi-instrument QQQ/XLK/NQ 5m sample
  scripts/                       # entry points (single pair, sweeps, baseline, multi-basket, data build)
  src/market_steiner/
    config.py                    # dataclass configs (data, signal, steiner, cost, basket)
    data/                        # CSV loader and preprocessing (log returns, alignment)
    mist/                        # optimizer.py (hedge ratio, raw hedge, convex update),
                                 #   graph.py (2-node activation), multinode.py (subset activation)
    strategy/                    # steiner_hedge.py, simple_hedge.py (baseline),
                                 #   multi_basket.py, costs.py
    backtest/                    # engine.py, metrics.py, param_sweep.py
  experimental/                  # earlier prototype, kept for reference
```

### experimental/

`experimental/` holds an earlier scratch version of this idea: flat scripts (`steiner_alpha_backtest.py`, `steiner_exec_sim.py`, `make_real_data.py`, `make_synthetic_data.py`, `run_steiner_alpha_demo.py`). The execution sim uses networkx Steiner-tree routines but is not wired into the main pipeline, and there are no saved results. This folder is superseded by the `src/` package and is kept only for reference.

## Results

The numbers below are read directly from `phase_sweep_results.csv`, the output of `scripts/run_phase_sweep.py`. The sweep covers a 3 x 3 x 4 grid over `lambda_etf` in {0.01, 0.02, 0.05}, `lambda_fut` in {0.02, 0.05, 0.08}, and `alpha` in {0.005, 0.01, 0.02, 0.05}, for 36 configurations, run on the included single-pair dataset (`data/raw/real_etf_fut_5m.csv`).

Annualized Sharpe across the grid:

- Best Sharpe: 2.160
- Worst Sharpe: 1.664
- Mean Sharpe across the 36 configs: 1.934

Best configuration:

| param | value |
|---|---|
| lambda_etf | 0.05 |
| lambda_fut | 0.02 |
| alpha | 0.01 |
| sharpe_annualized | 2.160 |
| turnover_notional_per_capital | 1.437 |
| total_pnl_net | 91.26 |
| total_costs | 103.08 |
| frac_etf_only | 0.364 |
| frac_fut_only | 0.574 |
| frac_both | 0.000 |
| frac_none | 0.061 |

Across every configuration in this sweep the BOTH activation is never selected (`frac_both = 0.0`); the strategy splits between ETF-only and future-only, with about 6.1% of bars in the no-trade band (`frac_none = 0.0614`). Net PnL is small relative to costs on this sample, so these Sharpe figures should be read as a property of this short in-sample dataset rather than a profitability claim.

## Limitations and notes

- This is a backtest only. There is no live trading, order routing, or broker integration.
- The included data is limited: the single-pair sample is 1536 5-minute bars (2025-09-17 to 2025-12-10) and the multi-instrument sample runs to 2025-12-10. Results will not generalize from this small window.
- The figures are in-sample backtest metrics on the bundled data, computed by the strategy's own cost and PnL model. They are not realized trading returns.
- The cost model is stylized (half-spread on traded notional plus optional commission and leakage), not a full execution or market-impact model.
- The multi-basket strategy currently sets transaction costs to zero in its PnL loop (see the TODO in `src/market_steiner/strategy/multi_basket.py`), so its metrics are gross of costs.
- Data files are checked in as small portfolio evidence. To regenerate them, run `python experimental/make_real_data.py` (single pair) or `python scripts/build_multi_csv.py` (multi-instrument); both download from yfinance and need network access.
