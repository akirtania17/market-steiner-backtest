from __future__ import annotations

from dataclasses import dataclass, field

from typing import List


@dataclass
class BasketInstrumentConfig:
    """
    One hedging instrument in a multi-ETF basket.

    name:        logical name (e.g. "QQQ", "XLK", "NQ")
    price_col:   column in the prices DataFrame (e.g. "qqq_mid")
    ret_col:     column in the returns DataFrame (e.g. "qqq_ret")
    lambda_leak: leakage / slippage cost parameter λ_i
    """
    name: str
    price_col: str
    ret_col: str
    lambda_leak: float


@dataclass
class BasketConfig:
    """
    Configuration for a multi-instrument ETF basket hedge.

    target_ret_col: which return series we're primarily hedging (e.g. sector ETF).
    instruments:    list of candidate hedging instruments (ETFs, futures, etc.).
    """
    target_ret_col: str = "target_ret"
    instruments: List[BasketInstrumentConfig] = field(default_factory=list)

@dataclass
class DataConfig:
    """
    Configuration for loading & preprocessing ETF–FUT data.
    """
    csv_path: str = "data/raw/real_etf_fut_5m.csv"
    etf_col: str = "etf_mid"
    fut_col: str = "fut_mid"
    timestamp_col: str = "timestamp"
    tz: str | None = None  # e.g. "UTC" if you want to localize


@dataclass
class SignalConfig:
    """
    Configuration for signal / hedge target construction.
    """
    # Rolling window (bars) used to estimate ETF–FUT relationship
    hedge_reg_window: int = 200

    # Small ridge term in the hedge regression to avoid degeneracy
    ridge_eps: float = 1e-6


@dataclass
class SteinerConfig:
    """
    Parameters for the Steiner activation and hedge optimization.
    """
    # Leakage-like penalties for ETF and FUT (dimensionless, per unit of notional change)
    lambda_etf: float = 0.05
    lambda_fut: float = 0.02

    # Steiner activation graph hyperparameters
    alpha: float = 0.01  # weight on distance between raw target and candidate
    beta: float = 0.5   # weight on leakage in activation cost
    gamma: float = 50.0  # not used in simple 2-node case but kept for extension

    # Smoothness / trading penalties in final hedge optimization
    k_l2: float = 1e-3       # L2 penalty on Δw
    lambda_l1: float = 0.0   # L1 penalty on Δw (can be approximated / extended)

    # No-trade band on raw hedge magnitude to cut tiny churn
    min_notional_threshold: float = 50.0


@dataclass
class CostConfig:
    """
    Parameters for the trading cost model.
    """
    initial_capital: float = 1_000_000.0

    # Half-spread (as fraction of price) for ETF & FUT.
    # E.g. 0.0001 ~ 1 bps half-spread.
    half_spread_etf: float = 0.0001
    half_spread_fut: float = 0.00005

    # Additional per-unit leakage / decay costs per bar
    leakage_per_bar_etf: float = 0.0
    leakage_per_bar_fut: float = 0.0

    # Commission per trade (per unit notional); set to 0 to ignore
    commission_per_notional: float = 0.0


@dataclass
class BacktestConfig:
    """
    Backtest-level configuration.
    """
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    steiner: SteinerConfig = field(default_factory=SteinerConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    # Bars per year for 5-minute data: ~ 252 trading days * 78 bars/day ≈ 19656
    bars_per_year: int = 252 * 78
