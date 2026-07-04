import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class GarchFitResult:
    order: Tuple[int, int]
    params: Dict[str, float]
    conditional_variance_pct2: np.ndarray
    success: bool
    message: str
    neg_loglik: float


def _conditional_variance(eps: np.ndarray, params: np.ndarray, p: int, q: int) -> np.ndarray:
    """Return conditional variance for GARCH(p, q). eps is in percent units."""
    omega = params[0]
    alphas = params[1 : 1 + p]
    betas = params[1 + p : 1 + p + q]
    t = len(eps)
    h = np.full(t, np.var(eps) if np.var(eps) > 1e-12 else 1e-6)
    max_lag = max(p, q)
    for i in range(max_lag, t):
        arch_term = 0.0
        garch_term = 0.0
        for a in range(p):
            arch_term += alphas[a] * eps[i - a - 1] ** 2
        for b in range(q):
            garch_term += betas[b] * h[i - b - 1]
        h[i] = omega + arch_term + garch_term
        if not np.isfinite(h[i]) or h[i] <= 1e-12:
            h[i] = 1e-12
    return h


def fit_garch(returns_decimal: np.ndarray, p: int, q: int) -> GarchFitResult:
    """
    Fit a simple Gaussian GARCH(p,q) model using scipy.
    Returns are decimals, e.g. 0.01 for 1%; internally scaled to percent for numerical stability.
    """
    r = np.asarray(returns_decimal, dtype=float)
    r = r[np.isfinite(r)]
    eps = (r - np.mean(r)) * 100.0
    n = len(eps)
    if n < max(p, q) + 20:
        raise ValueError(f"Not enough observations to fit GARCH({p},{q}).")

    var = np.var(eps)
    if var <= 1e-12:
        var = 1e-6

    # Conservative stationary initialization.
    alpha_total = 0.08
    beta_total = 0.86
    omega0 = max(var * (1.0 - alpha_total - beta_total), 1e-6)
    x0 = np.r_[omega0, np.repeat(alpha_total / p, p), np.repeat(beta_total / q, q)]
    bounds = [(1e-10, None)] + [(1e-10, 0.999)] * (p + q)

    def objective(x: np.ndarray) -> float:
        omega = x[0]
        alphas = x[1 : 1 + p]
        betas = x[1 + p :]
        persistence = np.sum(alphas) + np.sum(betas)
        if omega <= 0 or np.any(alphas < 0) or np.any(betas < 0) or persistence >= 0.999:
            return 1e10 + 1e9 * max(0.0, persistence - 0.999) ** 2
        h = _conditional_variance(eps, x, p, q)
        max_lag = max(p, q)
        h_use = h[max_lag:]
        eps_use = eps[max_lag:]
        if np.any(~np.isfinite(h_use)) or np.any(h_use <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(h_use) + eps_use**2 / h_use))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})

    params = res.x if res.success else x0
    h = _conditional_variance(eps, params, p, q)
    names = ["omega"] + [f"alpha{i}" for i in range(1, p + 1)] + [f"beta{j}" for j in range(1, q + 1)]
    return GarchFitResult(
        order=(p, q),
        params=dict(zip(names, params)),
        conditional_variance_pct2=h,
        success=bool(res.success),
        message=str(res.message),
        neg_loglik=float(res.fun if np.isfinite(res.fun) else np.nan),
    )
