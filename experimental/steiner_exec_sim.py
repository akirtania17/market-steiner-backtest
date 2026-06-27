from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from networkx import Graph
from networkx.algorithms.approximation import steiner_tree
from tqdm.auto import tqdm


# -------------------------
# Core data structures
# -------------------------


@dataclass
class AssetInfo:
    ticker: str
    is_proxy: bool
    sector_beta: float
    market_beta: float
    adv: float            # average daily volume (notional)
    volatility: float     # daily vol
    k: float              # impact coefficient
    alpha: float          # impact exponent
    spread_bps: float


@dataclass
class Universe:
    tickers: List[str]
    assets: Dict[str, AssetInfo]
    corr_matrix: np.ndarray
    etf_mapping: Dict[str, Dict[str, float]]   # ETF -> {stock: weight}
    fut_mapping: Dict[str, Dict[str, float]]   # Future -> {stock: weight}
    lambda_etf_leakage: float
    lambda_fut_leakage: float


# -------------------------
# Impact / cost helpers
# -------------------------


def impact_cost_single(asset: AssetInfo, q_notional: float) -> float:
    """
    Simple power-law impact cost:
        cost = k * |q|^alpha
    where q is notional traded.
    """
    if q_notional == 0.0:
        return 0.0
    return float(asset.k * (abs(q_notional) ** asset.alpha))


def etf_leakage_cost(universe: Universe, etf_ticker: str, q_etf: float) -> float:
    """
    Cost of ETF leakage into underlying stocks.
    """
    if q_etf == 0.0:
        return 0.0
    lam = universe.lambda_etf_leakage
    weights = universe.etf_mapping.get(etf_ticker, {})
    total_cost = 0.0
    for stock, w in weights.items():
        info = universe.assets[stock]
        q_stock = lam * w * q_etf
        total_cost += impact_cost_single(info, q_stock)
    return total_cost


def futures_leakage_cost(universe: Universe, fut_ticker: str, q_fut: float) -> float:
    """
    Cost of future leakage into underlying stocks.
    """
    if q_fut == 0.0:
        return 0.0
    lam = universe.lambda_fut_leakage
    weights = universe.fut_mapping.get(fut_ticker, {})
    total_cost = 0.0
    for stock, w in weights.items():
        info = universe.assets[stock]
        q_stock = lam * w * q_fut
        total_cost += impact_cost_single(info, q_stock)
    return total_cost


def compute_total_cost(universe: Universe, trades: Dict[str, float]) -> float:
    """
    Unified objective J(trades) used by all strategies.
    Sum of impact costs on each instrument plus leakage costs for proxies.
    """
    total = 0.0
    for t, q in trades.items():
        if abs(q) == 0.0:
            continue
        info = universe.assets[t]
        if info.is_proxy:
            # Direct impact
            total += impact_cost_single(info, q)
            # Leakage into underlyings
            if t in universe.etf_mapping:
                total += etf_leakage_cost(universe, t, q)
            if t in universe.fut_mapping:
                total += futures_leakage_cost(universe, t, q)
        else:
            total += impact_cost_single(info, q)
    return float(total)


# -------------------------
# Universe construction
# -------------------------


def make_random_corr(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    Dinv = np.diag(1.0 / d)
    corr = Dinv @ cov @ Dinv
    np.fill_diagonal(corr, 1.0)
    return corr


def load_simulated_universe(
    n_stocks: int,
    lambda_etf_leakage: float,
    lambda_fut_leakage: float,
    seed: int = 1234,
) -> Universe:
    rng = np.random.default_rng(seed)

    tickers: List[str] = []
    assets: Dict[str, AssetInfo] = {}

    # Generate stocks
    base_adv = 5e6
    base_vol = 0.02

    for i in range(n_stocks):
        t = f"STK{i:02d}"
        tickers.append(t)
        adv = float(base_adv * rng.lognormal(mean=0.0, sigma=0.5))
        vol = float(base_vol * rng.lognormal(mean=0.0, sigma=0.3))
        k = 7e-8 * (base_adv / adv)  # smaller ADV -> larger k
        alpha = 1.2
        spread_bps = float(rng.uniform(1.0, 4.0))
        sector_beta = float(rng.normal(loc=1.0, scale=0.2))
        market_beta = float(rng.normal(loc=1.0, scale=0.1))

        assets[t] = AssetInfo(
            ticker=t,
            is_proxy=False,
            sector_beta=sector_beta,
            market_beta=market_beta,
            adv=adv,
            volatility=vol,
            k=k,
            alpha=alpha,
            spread_bps=spread_bps,
        )

    # Correlation matrix for stocks
    corr = make_random_corr(n_stocks, seed + 1)

    # Sort stocks by ADV (largest first)
    stock_tickers = [t for t in tickers]
    stock_tickers.sort(key=lambda x: assets[x].adv, reverse=True)

    # --- ETF: exposure to top K stocks ---
    etf_ticker = "XLK"
    tickers.append(etf_ticker)
    K_etf = min(12, n_stocks)
    etf_underlying = stock_tickers[:K_etf]
    w = np.array(rng.lognormal(mean=0.0, sigma=0.3, size=K_etf))
    w = w / w.sum()
    etf_weights = {t: float(wi) for t, wi in zip(etf_underlying, w)}

    k_stocks = np.array([assets[t].k for t in etf_underlying])
    vol_stocks = np.array([assets[t].volatility for t in etf_underlying])

    etf_info = AssetInfo(
        ticker=etf_ticker,
        is_proxy=True,
        sector_beta=1.0,
        market_beta=1.25,
        adv=float(np.sum([assets[t].adv for t in etf_underlying]) * 1.0),
        volatility=float(vol_stocks.mean()),
        k=float(k_stocks.min() * 0.08),
        alpha=1.2,
        spread_bps=2.0,
    )
    assets[etf_ticker] = etf_info

    # --- Future: also on the sector, but very liquid & cheap ---
    fut_ticker = "NQ"
    tickers.append(fut_ticker)
    K_fut = min(20, n_stocks)
    fut_underlying = stock_tickers[:K_fut]
    wf = np.array(rng.lognormal(mean=0.0, sigma=0.3, size=K_fut))
    wf = wf / wf.sum()
    fut_weights = {t: float(wi) for t, wi in zip(fut_underlying, wf)}

    fut_info = AssetInfo(
        ticker=fut_ticker,
        is_proxy=True,
        sector_beta=1.0,
        market_beta=0.9,
        adv=float(np.sum([assets[t].adv for t in fut_underlying]) * 2.0),
        volatility=float(vol_stocks.mean()),
        k=float(k_stocks.min() * 0.35),
        alpha=1.1,
        spread_bps=1.0,
    )
    assets[fut_ticker] = fut_info

    # Extend corr matrix to include proxies as uncorrelated
    n = n_stocks
    corr_ext = np.eye(n + 2)
    corr_ext[:n, :n] = corr

    etf_mapping = {etf_ticker: etf_weights}
    fut_mapping = {fut_ticker: fut_weights}

    return Universe(
        tickers=tickers,
        assets=assets,
        corr_matrix=corr_ext,
        etf_mapping=etf_mapping,
        fut_mapping=fut_mapping,
        lambda_etf_leakage=lambda_etf_leakage,
        lambda_fut_leakage=lambda_fut_leakage,
    )


# -------------------------
# Holdings generator
# -------------------------


def generate_random_holdings(universe: Universe, rng: np.random.Generator) -> Dict[str, float]:
    holdings: Dict[str, float] = {}
    for t, info in universe.assets.items():
        if info.is_proxy:
            holdings[t] = 0.0
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            holdings[t] = float(sign * info.adv * rng.uniform(0.1, 0.5))
    return holdings


def compute_sector_exposure(universe: Universe, holdings: Dict[str, float]) -> float:
    sector_exp = 0.0
    for t, info in universe.assets.items():
        if info.is_proxy:
            continue
        h = holdings.get(t, 0.0)
        sector_exp += h * info.sector_beta
    return float(sector_exp)


# -------------------------
# Strategies (trades only)
# -------------------------


def baseline_per_name_trades(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
) -> Dict[str, float]:
    """
    Trade each stock directly to reduce sector exposure by a fraction,
    proportionally across names.
    """
    trades = {t: 0.0 for t in universe.tickers}
    stock_tickers = [t for t, info in universe.assets.items() if not info.is_proxy]

    # Compute total sector exposure
    sector_exp = compute_sector_exposure(universe, current_holdings)
    desired_change = sector_reduction * sector_exp  # ΔS

    # Proportional trade: Δh_i ~ sector_beta_i * h_i
    denom = 0.0
    for t in stock_tickers:
        h = current_holdings.get(t, 0.0)
        info = universe.assets[t]
        denom += (info.sector_beta ** 2) * abs(h)

    if denom == 0.0:
        return trades

    for t in stock_tickers:
        h = current_holdings.get(t, 0.0)
        info = universe.assets[t]
        weight = (info.sector_beta ** 2) * abs(h) / denom
        delta_exposure = desired_change * weight
        beta = info.sector_beta if info.sector_beta != 0 else 1.0
        q_notional = delta_exposure / beta
        trades[t] = q_notional

    return trades


def factor_etf_only_trades(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
    etf_notional_cap: float,
) -> Dict[str, float]:
    """
    Simple factor strategy: use ETF only to achieve desired sector change.
    """
    trades = {t: 0.0 for t in universe.tickers}
    stock_tickers = [t for t, info in universe.assets.items() if not info.is_proxy]

    sector_exp = compute_sector_exposure(universe, current_holdings)
    desired_change = sector_reduction * sector_exp  # ΔS

    etf_ticker: Optional[str] = None
    for t, info in universe.assets.items():
        if info.is_proxy and t in universe.etf_mapping:
            etf_ticker = t
            break

    if etf_ticker is None:
        # Fall back to baseline
        return baseline_per_name_trades(universe, current_holdings, sector_reduction)

    etf_info = universe.assets[etf_ticker]
    beta_S = etf_info.sector_beta if etf_info.sector_beta != 0 else 1.0

    qT_unclipped = desired_change / beta_S
    qT = float(np.clip(qT_unclipped, -etf_notional_cap, etf_notional_cap))
    trades[etf_ticker] = qT

    return trades


def proxy_only_trades(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
    proxy_ticker: str,
    proxy_cap: float,
) -> Dict[str, float]:
    """
    Helper: trades using only a single proxy (ETF or FUT).
    """
    trades = {t: 0.0 for t in universe.tickers}
    sector_exp = compute_sector_exposure(universe, current_holdings)
    desired_change = sector_reduction * sector_exp

    info_p = universe.assets[proxy_ticker]
    beta_S = info_p.sector_beta if info_p.sector_beta != 0 else 1.0

    q_unclipped = desired_change / beta_S
    q = float(np.clip(q_unclipped, -proxy_cap, proxy_cap))
    trades[proxy_ticker] = q
    return trades


# -------------------------
# Steiner graph (benefit-based)
# -------------------------


def build_execution_graph_benefit_based(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
    etf_notional_cap: float,
    fut_notional_cap: float,
    max_terminal_stocks: int = 8,
) -> Tuple[Graph, List[str], List[str], float]:
    """
    Build a benefit-based graph for Steiner.

    Idea:
      - Compute J_base (stocks only)
      - For each proxy, compute J_proxy (that proxy only)
      - Edge ROOT--proxy has weight ~ 1 / max(J_base - J_proxy, eps)
        => cheaper edge if proxy gives large improvement vs baseline.

    Nodes:
      - ROOT
      - proxy nodes (ETF, futures)
      - top terminal stocks by |holding * beta|

    Returns:
      G, proxies, terminal_stocks, sector_exp
    """
    G: Graph = nx.Graph()
    ROOT = "ROOT"
    G.add_node(ROOT)

    proxies = [t for t, info in universe.assets.items() if info.is_proxy]
    stocks = [t for t, info in universe.assets.items() if not info.is_proxy]

    # Sector exposure (for info)
    sector_exp = compute_sector_exposure(universe, current_holdings)

    # Baseline trades/cost
    base_trades = baseline_per_name_trades(universe, current_holdings, sector_reduction)
    J_base = compute_total_cost(universe, base_trades)

    # Pick terminal stocks by |holding * sector_beta|
    scored: List[Tuple[float, str]] = []
    for t in stocks:
        h = current_holdings.get(t, 0.0)
        info = universe.assets[t]
        score = abs(h * info.sector_beta)
        scored.append((score, t))
    scored.sort(reverse=True)
    terminal_stocks = [t for _, t in scored[:max_terminal_stocks]]

    for t in proxies + terminal_stocks:
        G.add_node(t)

    # Proxy capacities
    etf_ticker = next((t for t in proxies if t in universe.etf_mapping), None)
    fut_ticker = next((t for t in proxies if t in universe.fut_mapping), None)

    # ROOT <-> proxy edges based on benefit vs baseline
    eps = 1e-9
    for p in proxies:
        if p == etf_ticker:
            cap = etf_notional_cap
        elif p == fut_ticker:
            cap = fut_notional_cap
        else:
            cap = 0.0
        if cap == 0.0:
            continue

        trades_p = proxy_only_trades(universe, current_holdings, sector_reduction, p, cap)
        J_p = compute_total_cost(universe, trades_p)
        benefit = max(J_base - J_p, 0.0)
        weight = 1.0 / (benefit + eps)
        # small additive term to avoid zero
        G.add_edge(ROOT, p, weight=float(weight + 1e-6))

    # proxy <-> stock edges as "light" connections so the tree spans stocks
    corr = universe.corr_matrix
    all_tickers = universe.tickers
    idx_map = {t: i for i, t in enumerate(all_tickers)}

    # Use a small correlation-based cost from proxy to stocks that are in its mapping
    for p in proxies:
        if p in universe.etf_mapping:
            for stock in universe.etf_mapping[p].keys():
                if stock not in terminal_stocks:
                    continue
                ii = idx_map.get(stock, None)
                jj = idx_map.get(stock, None)
                if ii is None or jj is None:
                    continue
                # Very small edge: we mainly care that the tree can span stocks via p
                G.add_edge(p, stock, weight=1e-3)
        if p in universe.fut_mapping:
            for stock in universe.fut_mapping[p].keys():
                if stock not in terminal_stocks:
                    continue
                G.add_edge(p, stock, weight=1e-3)

    # stock <-> stock edges: correlation distance to keep tree nicely connected
    for i, si in enumerate(terminal_stocks):
        for sj in terminal_stocks[i + 1 :]:
            ii = idx_map[si]
            jj = idx_map[sj]
            rho = corr[ii, jj]
            dist = 1.0 - abs(rho)
            G.add_edge(si, sj, weight=float(dist + 1e-3))

    return G, proxies, terminal_stocks, sector_exp


def steiner_strategy(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
    etf_notional_cap: float,
    fut_notional_cap: float,
    n_grid: int = 101,
) -> Tuple[Dict[str, float], Graph, List[str]]:
    """
    Steiner-style strategy, but objective is unified with the convex "optimal".

    1. Build benefit-based graph.
    2. Use Steiner tree to pick active proxies.
    3. For the selected proxies, run a low-dimensional grid search on notional
       (q_ETF, q_FUT) under exposure + cap constraints to minimize true J.
    4. Return trades, tree, and active proxies (for diagnostics).
    """
    trades = {t: 0.0 for t in universe.tickers}

    G, proxies, terminal_stocks, sector_exp = build_execution_graph_benefit_based(
        universe,
        current_holdings,
        sector_reduction,
        etf_notional_cap,
        fut_notional_cap,
    )

    ROOT = "ROOT"
    terminals = set(terminal_stocks)
    terminals.add(ROOT)

    if len(G.nodes) == 1:
        # Only ROOT
        return trades, G, []

    T = steiner_tree(G, terminals, weight="weight")
    active_proxies = [p for p in proxies if p in T.nodes]

    # Sector exposure and desired change
    sector_exp = compute_sector_exposure(universe, current_holdings)
    desired_change = sector_reduction * sector_exp

    # Identify ETF / FUT among active proxies
    etf_candidates = [p for p in active_proxies if p in universe.etf_mapping]
    fut_candidates = [p for p in active_proxies if p in universe.fut_mapping]

    # Helper to evaluate cost for specific proxy notionals
    def eval_cost(q_etf: float, q_fut: float) -> Tuple[float, Dict[str, float]]:
        local_trades = {t: 0.0 for t in universe.tickers}
        if etf_candidates:
            local_trades[etf_candidates[0]] = q_etf
        if fut_candidates:
            local_trades[fut_candidates[0]] = q_fut
        J = compute_total_cost(universe, local_trades)
        return J, local_trades

    # No active proxies → fallback to baseline
    if not etf_candidates and not fut_candidates:
        base_trades = baseline_per_name_trades(universe, current_holdings, sector_reduction)
        return base_trades, T, []

    # Case 1: only one proxy in tree
    if etf_candidates and not fut_candidates:
        p = etf_candidates[0]
        beta = universe.assets[p].sector_beta or 1.0
        q_unclipped = desired_change / beta
        q = float(np.clip(q_unclipped, -etf_notional_cap, etf_notional_cap))
        trades[p] = q
        return trades, T, etf_candidates

    if fut_candidates and not etf_candidates:
        p = fut_candidates[0]
        beta = universe.assets[p].sector_beta or 1.0
        q_unclipped = desired_change / beta
        q = float(np.clip(q_unclipped, -fut_notional_cap, fut_notional_cap))
        trades[p] = q
        return trades, T, fut_candidates

    # Case 2: both ETF and FUT active.
    etf_ticker = etf_candidates[0]
    fut_ticker = fut_candidates[0]
    beta_T = universe.assets[etf_ticker].sector_beta or 1.0
    beta_F = universe.assets[fut_ticker].sector_beta or 1.0

    best_J = float("inf")
    best_trades: Dict[str, float] = {t: 0.0 for t in universe.tickers}

    # Grid over q_ETF; solve for q_FUT from exposure constraint
    min_qT = -etf_notional_cap
    max_qT = etf_notional_cap

    for i in range(n_grid):
        qT = min_qT + (max_qT - min_qT) * i / (n_grid - 1)
        # Exposure: beta_T * qT + beta_F * qF = desired_change
        qF = (desired_change - beta_T * qT) / (beta_F + 1e-12)
        if abs(qF) > fut_notional_cap:
            continue
        J, local_trades = eval_cost(qT, qF)
        if J < best_J:
            best_J = J
            best_trades = local_trades

    return best_trades, T, active_proxies


# -------------------------
# Optimal ETF + FUT strategy
# -------------------------


def optimal_two_proxy_strategy(
    universe: Universe,
    current_holdings: Dict[str, float],
    sector_reduction: float,
    etf_notional_cap: float,
    fut_notional_cap: float,
    n_grid: int = 201,
) -> Dict[str, float]:
    """
    "Ground truth" optimal within ETF + FUT space using the SAME objective J.

    We:
      - Always allow both XLK and NQ (if present).
      - Grid over q_ETF, solve q_FUT from exposure constraint.
      - Enforce caps and pick the minimum J.
    """
    trades = {t: 0.0 for t in universe.tickers}
    proxies = [t for t, info in universe.assets.items() if info.is_proxy]
    etf_ticker = next((t for t in proxies if t in universe.etf_mapping), None)
    fut_ticker = next((t for t in proxies if t in universe.fut_mapping), None)

    if etf_ticker is None and fut_ticker is None:
        # No proxies: baseline is "optimal"
        return baseline_per_name_trades(universe, current_holdings, sector_reduction)

    sector_exp = compute_sector_exposure(universe, current_holdings)
    desired_change = sector_reduction * sector_exp

    # Helper: evaluate cost for given qT, qF
    def eval_cost(qT: float, qF: float) -> Tuple[float, Dict[str, float]]:
        local_trades = {t: 0.0 for t in universe.tickers}
        if etf_ticker is not None:
            local_trades[etf_ticker] = qT
        if fut_ticker is not None:
            local_trades[fut_ticker] = qF
        J = compute_total_cost(universe, local_trades)
        return J, local_trades

    # If only one proxy exists, this degenerates.
    if etf_ticker is None:
        beta_F = universe.assets[fut_ticker].sector_beta or 1.0
        q_unclipped = desired_change / beta_F
        q = float(np.clip(q_unclipped, -fut_notional_cap, fut_notional_cap))
        trades[fut_ticker] = q
        return trades

    if fut_ticker is None:
        beta_T = universe.assets[etf_ticker].sector_beta or 1.0
        q_unclipped = desired_change / beta_T
        q = float(np.clip(q_unclipped, -etf_notional_cap, etf_notional_cap))
        trades[etf_ticker] = q
        return trades

    beta_T = universe.assets[etf_ticker].sector_beta or 1.0
    beta_F = universe.assets[fut_ticker].sector_beta or 1.0

    best_J = float("inf")
    best_trades = {t: 0.0 for t in universe.tickers}

    min_qT = -etf_notional_cap
    max_qT = etf_notional_cap

    for i in range(n_grid):
        qT = min_qT + (max_qT - min_qT) * i / (n_grid - 1)
        qF = (desired_change - beta_T * qT) / (beta_F + 1e-12)
        if abs(qF) > fut_notional_cap:
            continue
        J, local_trades = eval_cost(qT, qF)
        if J < best_J:
            best_J = J
            best_trades = local_trades

    return best_trades


# -------------------------
# Diagnostics on worst scenarios
# -------------------------


def analyze_worst_scenarios_for_lambda(
    lambda_etf: float,
    scenario_results: List[Dict],
    ratio_threshold: float = 1.05,
    max_show: int = 3,
) -> None:
    """
    Look at scenarios where Steiner is significantly worse than Optimal
    and print detailed diagnostics.
    """
    ratios = []
    for res in scenario_results:
        J_opt = res["J_opt"]
        J_stein = res["J_stein"]
        if J_opt <= 0:
            continue
        r = J_stein / J_opt
        ratios.append((r, res))

    bad = [(r, res) for r, res in ratios if r > ratio_threshold]
    bad.sort(key=lambda x: x[0], reverse=True)

    print()
    print(f"=== Detailed analysis for λ_etf = {lambda_etf:.2f} ===")
    print()
    print(
        f"Found {len(bad)} scenarios with Steiner/Optimal > {ratio_threshold:.2f} "
        f"out of {len(scenario_results)} (λ_etf={lambda_etf:.2f})."
    )
    if not bad:
        print("Showing up to 3 worst examples.")
        return

    print("Showing up to 3 worst examples.")
    print()

    for idx, (ratio, res) in enumerate(bad[:max_show]):
        J_base = res["J_base"]
        J_fact = res["J_fact"]
        J_stein = res["J_stein"]
        J_opt = res["J_opt"]
        scenario_idx = res["scenario_idx"]
        desired_change = res["desired_change"]
        etf_cap = res["etf_cap"]
        fut_cap = res["fut_cap"]
        trades_stein = res["trades_stein"]
        trades_opt = res["trades_opt"]
        T_stein: Graph = res["stein_tree"]
        universe: Universe = res["universe"]

        print("=" * 80)
        print(
            f"Scenario {scenario_idx}  |  λ_etf={lambda_etf:.2f}, "
            f"Steiner/Optimal = {ratio:.3f}"
        )
        print(
            f"Objectives:  J_base={J_base:.2f},  J_fact={J_fact:.2f}, "
            f"J_stein={J_stein:.2f},  J_opt={J_opt:.2f}"
        )
        print(
            f"Desired sector change ΔS = {desired_change: .2e},  "
            f"ETF cap={etf_cap: .2e}, FUT cap={fut_cap: .2e}"
        )
        print()

        def format_trades(trades: Dict[str, float], label: str) -> None:
            items = sorted(
                trades.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            print(f"Top {label} trades (by |notional|):")
            for t, q in items[:10]:
                if abs(q) < 1e-6:
                    continue
                info = universe.assets[t]
                kind = "ETF/FUT" if info.is_proxy else "STOCK"
                print(f"  {t:6s} {q: .3e}  [{kind}]")
            print()

        format_trades(trades_stein, "Steiner")
        format_trades(trades_opt, "Optimal")

        # Steiner tree structure
        print("Steiner tree structure:")
        print(f"  Nodes: {list(T_stein.nodes())}")
        active_proxies = [
            t
            for t in T_stein.nodes()
            if t in universe.assets and universe.assets[t].is_proxy
        ]
        print(f"  Active proxies in tree: {active_proxies}")
        print("  Edges (u, v, weight):")
        for u, v, data in T_stein.edges(data=True):
            w = data.get("weight", 0.0)
            print(f"    {u:6s} -- {v:6s}  w={w: .3e}")
        print()

    print()


# -------------------------
# Experiment driver
# -------------------------


def run_experiment_for_lambda(
    lambda_etf_leakage: float,
    lambda_fut_leakage: float,
    n_scenarios: int,
    sector_reduction: float,
    base_seed: int = 123,
) -> Tuple[List[Dict], Dict[str, float]]:
    rng = np.random.default_rng(base_seed)

    universe = load_simulated_universe(
        n_stocks=25,
        lambda_etf_leakage=lambda_etf_leakage,
        lambda_fut_leakage=lambda_fut_leakage,
        seed=base_seed,
    )

    scenario_results: List[Dict] = []

    for scenario_idx in range(n_scenarios):
        holdings = generate_random_holdings(universe, rng)

        total_notional = sum(
            abs(v) for t, v in holdings.items() if not universe.assets[t].is_proxy
        )
        etf_cap = 0.25 * total_notional
        fut_cap = 0.25 * total_notional

        # Get trades
        trades_base = baseline_per_name_trades(universe, holdings, sector_reduction)
        trades_fact = factor_etf_only_trades(universe, holdings, sector_reduction, etf_cap)
        trades_stein, T_stein, active_proxies = steiner_strategy(
            universe, holdings, sector_reduction, etf_cap, fut_cap
        )
        trades_opt = optimal_two_proxy_strategy(
            universe, holdings, sector_reduction, etf_cap, fut_cap
        )

        # Unified objectives
        J_base = compute_total_cost(universe, trades_base)
        J_fact = compute_total_cost(universe, trades_fact)
        J_stein = compute_total_cost(universe, trades_stein)
        J_opt = compute_total_cost(universe, trades_opt)

        # Sector exposure/change (for diagnostics)
        sector_exp = compute_sector_exposure(universe, holdings)
        desired_change = sector_reduction * sector_exp

        scenario_results.append(
            {
                "scenario_idx": scenario_idx,
                "universe": universe,
                "holdings": holdings,
                "trades_base": trades_base,
                "trades_fact": trades_fact,
                "trades_stein": trades_stein,
                "trades_opt": trades_opt,
                "J_base": J_base,
                "J_fact": J_fact,
                "J_stein": J_stein,
                "J_opt": J_opt,
                "stein_tree": T_stein,
                "stein_active_proxies": active_proxies,
                "desired_change": desired_change,
                "etf_cap": etf_cap,
                "fut_cap": fut_cap,
            }
        )

    # Aggregate stats
    J_base_arr = np.array([res["J_base"] for res in scenario_results])
    J_fact_arr = np.array([res["J_fact"] for res in scenario_results])
    J_stein_arr = np.array([res["J_stein"] for res in scenario_results])
    J_opt_arr = np.array([res["J_opt"] for res in scenario_results])

    ratio_base_stein = J_base_arr / J_stein_arr
    ratio_fact_stein = J_fact_arr / J_stein_arr

    ratio_base_opt = J_base_arr / J_opt_arr
    ratio_fact_opt = J_fact_arr / J_opt_arr
    ratio_stein_opt = J_stein_arr / J_opt_arr

    pct_stein_lt_fact = float((J_stein_arr < J_fact_arr).mean())
    pct_stein_lt_base = float((J_stein_arr < J_base_arr).mean())
    pct_stein_within_10_opt = float((ratio_stein_opt <= 1.10).mean())

    # Proxy usage stats
    universe_assets = universe.assets
    def used_proxy(res_list: List[Dict], kind: str) -> float:
        count = 0
        for r in res_list:
            trades = r["trades_stein"]
            for t, q in trades.items():
                if abs(q) < 1e-6:
                    continue
                info = universe_assets[t]
                if not info.is_proxy:
                    continue
                if kind == "ETF" and t in universe.etf_mapping:
                    count += 1
                    break
                if kind == "FUT" and t in universe.fut_mapping:
                    count += 1
                    break
        return count / len(res_list)

    pct_etf_used = used_proxy(scenario_results, "ETF")
    pct_fut_used = used_proxy(scenario_results, "FUT")

    # Print summary
    print("=== Experiment summary ===")
    print(f"lambda_etf_leakage: {lambda_etf_leakage}")
    print(f"lambda_fut_leakage: {lambda_fut_leakage}")
    print(f"Scenarios: {n_scenarios}")
    print(f"Avg objective baseline : {J_base_arr.mean()}")
    print(f"Avg objective factor   : {J_fact_arr.mean()}")
    print(f"Avg objective steiner  : {J_stein_arr.mean()}")
    print(f"Avg objective optimal  : {J_opt_arr.mean()}")
    print()
    print(
        "Baseline / Steiner ratio stats: "
        f"mean={ratio_base_stein.mean():.3f}, "
        f"median={np.median(ratio_base_stein):.3f}, "
        f"p10={np.percentile(ratio_base_stein, 10):.3f}, "
        f"p90={np.percentile(ratio_base_stein, 90):.3f}"
    )
    print(
        "Factor   / Steiner ratio stats: "
        f"mean={ratio_fact_stein.mean():.3f}, "
        f"median={np.median(ratio_fact_stein):.3f}, "
        f"p10={np.percentile(ratio_fact_stein, 10):.3f}, "
        f"p90={np.percentile(ratio_fact_stein, 90):.3f}"
    )
    print()
    print(
        "Baseline / Optimal ratio stats: "
        f"mean={ratio_base_opt.mean():.3f}, "
        f"median={np.median(ratio_base_opt):.3f}, "
        f"p10={np.percentile(ratio_base_opt, 10):.3f}, "
        f"p90={np.percentile(ratio_base_opt, 90):.3f}"
    )
    print(
        "Factor   / Optimal ratio stats: "
        f"mean={ratio_fact_opt.mean():.3f}, "
        f"median={np.median(ratio_fact_opt):.3f}, "
        f"p10={np.percentile(ratio_fact_opt, 10):.3f}, "
        f"p90={np.percentile(ratio_fact_opt, 90):.3f}"
    )
    print(
        "Steiner  / Optimal ratio stats: "
        f"mean={ratio_stein_opt.mean():.3f}, "
        f"median={np.median(ratio_stein_opt):.3f}, "
        f"p10={np.percentile(ratio_stein_opt, 10):.3f}, "
        f"p90={np.percentile(ratio_stein_opt, 90):.3f}"
    )
    print()
    print(f"Pct scenarios Steiner < Factor  (J): {pct_stein_lt_fact}")
    print(f"Pct scenarios Steiner < Baseline (J): {pct_stein_lt_base}")
    print(f"Pct scenarios Steiner <= Optimal (J): {(ratio_stein_opt <= 1.0 + 1e-12).mean()}")
    print(f"Pct scenarios Steiner within 10% of Optimal (J): {pct_stein_within_10_opt}")
    print(f"Pct scenarios Steiner used ETF: {pct_etf_used:.3f}")
    print(f"Pct scenarios Steiner used FUT: {pct_fut_used:.3f}")
    print()

    # Return stats for further use
    summary = {
        "avg_J_base": J_base_arr.mean(),
        "avg_J_fact": J_fact_arr.mean(),
        "avg_J_stein": J_stein_arr.mean(),
        "avg_J_opt": J_opt_arr.mean(),
    }
    return scenario_results, summary


def run_lambda_sweep() -> None:
    n_scenarios = 50
    sector_reduction = -0.15
    lambda_fut_leakage = 0.08
    base_seed = 123

    all_results_by_lambda: Dict[float, List[Dict]] = {}

    lambdas = [0.1 * i for i in range(1, 10)]
    for lam in tqdm(lambdas, desc="λ sweep"):
        scenario_results, summary = run_experiment_for_lambda(
            lambda_etf_leakage=lam,
            lambda_fut_leakage=lambda_fut_leakage,
            n_scenarios=n_scenarios,
            sector_reduction=sector_reduction,
            base_seed=base_seed,
        )
        all_results_by_lambda[lam] = scenario_results

    # Detailed diagnostics for a chosen lambda, e.g. λ_etf = 0.7
    target_lambda = 0.7
    if target_lambda in all_results_by_lambda:
        analyze_worst_scenarios_for_lambda(
            target_lambda,
            all_results_by_lambda[target_lambda],
            ratio_threshold=1.05,   # 5% worse than optimal
            max_show=3,
        )


if __name__ == "__main__":
    run_lambda_sweep()
