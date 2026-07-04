from typing import Literal

import numpy as np
from scipy.special import gammaln

TreeType = Literal["crr", "eqp"]


def _tree_parameters(
    sigma: np.ndarray,
    r: np.ndarray,
    ttm: np.ndarray,
    steps: int,
    tree_type: TreeType = "crr",
    q: float = 0.0,
):
    dt = ttm / steps
    sqrt_dt = np.sqrt(dt)
    if tree_type == "crr":
        u = np.exp(sigma * sqrt_dt)
        d = 1.0 / u
        growth = np.exp((r - q) * dt)
        p = (growth - d) / (u - d)
    elif tree_type == "eqp":
        # Equal-probability / Jarrow-Rudd tree: p=0.5, u and d include the risk-neutral drift.
        drift = (r - q - 0.5 * sigma**2) * dt
        u = np.exp(drift + sigma * sqrt_dt)
        d = np.exp(drift - sigma * sqrt_dt)
        p = np.full_like(sigma, 0.5, dtype=float)
    else:
        raise ValueError("tree_type must be either 'crr' or 'eqp'.")
    return u, d, p, dt


def price_binomial_vectorized(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    steps: int = 100,
    tree_type: TreeType = "crr",
    q: float = 0.0,
    batch_size: int = 200000,
) -> np.ndarray:
    """
    Vectorized European option pricing by CRR or EQP/Jarrow-Rudd binomial tree.

    For speed, this uses the closed-form binomial distribution representation of
    terminal-state expectation instead of explicitly building the full tree:

        V0 = exp(-rT) E_Q[payoff(S_N)].

    This is equivalent to backward induction for European options and is much
    faster for large option panels.
    """
    from scipy.stats import binom

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    opt = np.asarray(option_type).astype(str)
    n = len(S)
    prices = np.full(n, np.nan, dtype=float)

    base_valid = (
        np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(sigma)
        & (S > 0) & (K > 0) & (T > 0) & (sigma > 1e-8)
    )
    if not np.any(base_valid):
        return prices

    valid_idx = np.flatnonzero(base_valid)
    for start in range(0, len(valid_idx), batch_size):
        idx = valid_idx[start : start + batch_size]
        Sb, Kb, Tb, rb, sigb = S[idx], K[idx], T[idx], r[idx], sigma[idx]
        u, d, p, dt = _tree_parameters(sigb, rb, Tb, steps, tree_type=tree_type, q=q)
        R = p * u + (1.0 - p) * d
        p_star = np.where(R > 0, p * u / R, np.nan)

        valid = (
            np.isfinite(u) & np.isfinite(d) & np.isfinite(p) & np.isfinite(R) & np.isfinite(p_star)
            & (u > 0) & (d > 0) & (u > d) & (R > 0) & (p_star >= 0) & (p_star <= 1)
        )
        if tree_type == "crr":
            valid &= (p > 0) & (p < 1)
        if not np.any(valid):
            continue

        local_idx = np.flatnonzero(valid)
        global_idx = idx[local_idx]
        Sb = Sb[local_idx]
        Kb = Kb[local_idx]
        Tb = Tb[local_idx]
        rb = rb[local_idx]
        u = u[local_idx]
        d = d[local_idx]
        p = p[local_idx]
        R = R[local_idx]
        p_star = p_star[local_idx]

        # Critical number of up moves for call moneyness: S*u^j*d^(N-j) >= K.
        denom = np.log(u / d)
        a = np.ceil((np.log(Kb / Sb) - steps * np.log(d)) / denom).astype(int)

        tail_p = np.empty_like(Sb, dtype=float)       # P(J >= a), J~Bin(N,p)
        tail_p_star = np.empty_like(Sb, dtype=float)  # P(J >= a), J~Bin(N,p_star)
        lower_p = np.empty_like(Sb, dtype=float)      # P(J < a)
        lower_p_star = np.empty_like(Sb, dtype=float) # P(J < a)

        below = a <= 0
        above = a > steps
        mid = ~(below | above)

        tail_p[below] = 1.0
        tail_p_star[below] = 1.0
        lower_p[below] = 0.0
        lower_p_star[below] = 0.0

        tail_p[above] = 0.0
        tail_p_star[above] = 0.0
        lower_p[above] = 1.0
        lower_p_star[above] = 1.0

        if np.any(mid):
            k = a[mid] - 1
            tail_p[mid] = binom.sf(k, steps, p[mid])
            tail_p_star[mid] = binom.sf(k, steps, p_star[mid])
            lower_p[mid] = binom.cdf(k, steps, p[mid])
            lower_p_star[mid] = binom.cdf(k, steps, p_star[mid])

        disc = np.exp(-rb * Tb)
        stock_factor = Sb * (R ** steps)
        call_price = disc * (stock_factor * tail_p_star - Kb * tail_p)
        put_price = disc * (Kb * lower_p - stock_factor * lower_p_star)
        is_call = np.char.upper(opt[global_idx]) == "C"
        prices[global_idx] = np.where(is_call, call_price, put_price)

    # Small numerical errors can make prices slightly negative near zero.
    prices = np.where(np.isfinite(prices), np.maximum(prices, 0.0), np.nan)
    return prices


def build_example_tree(S: float, T: float, r: float, sigma: float, steps: int, tree_type: TreeType, q: float = 0.0):
    """Return asset-price tree levels for visualization."""
    S_arr = np.array([S], dtype=float)
    T_arr = np.array([T], dtype=float)
    r_arr = np.array([r], dtype=float)
    sig_arr = np.array([sigma], dtype=float)
    u, d, p, dt = _tree_parameters(sig_arr, r_arr, T_arr, steps, tree_type=tree_type, q=q)
    u, d = float(u[0]), float(d[0])
    levels = []
    for i in range(steps + 1):
        vals = [S * (u**j) * (d ** (i - j)) for j in range(i + 1)]
        levels.append(vals)
    return levels
