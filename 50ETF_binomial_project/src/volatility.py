from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, HIST_VOL_WINDOWS, GARCH_ORDERS
from .garch import fit_garch


def add_historical_volatility(
    etf: pd.DataFrame,
    windows: List[int] = HIST_VOL_WINDOWS,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Add log returns and annualized rolling historical volatilities."""
    out = etf.sort_values("trade_date").copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for w in windows:
        out[f"hist_vol_{w}d"] = out["log_return"].rolling(w).std() * np.sqrt(trading_days)
    return out


def add_garch_volatility(
    etf: pd.DataFrame,
    orders: List[Tuple[int, int]] = GARCH_ORDERS,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit GARCH models and add annualized conditional volatilities."""
    out = etf.sort_values("trade_date").copy()
    if "log_return" not in out.columns:
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    ret = out["log_return"].dropna().values
    return_index = out.index[out["log_return"].notna()]
    fit_rows = []

    for p, q in orders:
        result = fit_garch(ret, p, q)
        col = f"garch_{p}_{q}_vol"
        out[col] = np.nan
        # h is variance in percent-squared. Convert to annual decimal volatility.
        annualized_vol = np.sqrt(result.conditional_variance_pct2) / 100.0 * np.sqrt(trading_days)
        out.loc[return_index, col] = annualized_vol
        row = {"model": f"GARCH({p},{q})", "success": result.success, "message": result.message, "neg_loglik": result.neg_loglik}
        row.update(result.params)
        fit_rows.append(row)

    return out, pd.DataFrame(fit_rows)


def compute_all_volatility(etf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all requested volatility columns."""
    vol = add_historical_volatility(etf)
    vol, garch_fit = add_garch_volatility(vol)
    return vol, garch_fit


def volatility_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("hist_vol_") or c.startswith("garch_") and c.endswith("_vol")]
