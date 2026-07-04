import numpy as np
import pandas as pd


def add_moneyness_and_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["moneyness"] = out["underlying_close"] / out["exercise_price"]

    is_call = out["call_put"].astype(str).str.upper().eq("C")
    m = out["moneyness"]
    out["moneyness_group"] = np.select(
        [
            is_call & (m > 1.03),
            is_call & (m < 0.97),
            (~is_call) & (m < 0.97),
            (~is_call) & (m > 1.03),
            m.notna(),
        ],
        ["ITM", "OTM", "ITM", "OTM", "ATM"],
        default="NA",
    )
    out["maturity_group"] = pd.cut(
        out["days_to_maturity"],
        bins=[-1, 30, 90, np.inf],
        labels=["short_0_30d", "middle_31_90d", "long_90d_plus"],
    )
    out["liquidity_group"] = pd.qcut(out["vol"].rank(method="first"), 3, labels=["low", "medium", "high"])
    return out


def add_pricing_errors(df: pd.DataFrame, price_cols: list[str], market_col: str = "market_price") -> pd.DataFrame:
    out = df.copy()
    market = out[market_col].astype(float)
    for col in price_cols:
        out[f"{col}_error"] = out[col] - market
        out[f"{col}_abs_error"] = (out[col] - market).abs()
        out[f"{col}_rel_error"] = np.where(market > 0, (out[col] - market) / market, np.nan)
    return out


def summarize_errors(df: pd.DataFrame, price_cols: list[str], market_col: str = "market_price") -> pd.DataFrame:
    rows = []
    for col in price_cols:
        err = df[col] - df[market_col]
        rel = np.where(df[market_col] > 0, err / df[market_col], np.nan)
        valid = np.isfinite(err) & np.isfinite(rel)
        rows.append(
            {
                "model": col,
                "n": int(valid.sum()),
                "mean_model_price": float(np.nanmean(df.loc[valid, col])),
                "mean_market_price": float(np.nanmean(df.loc[valid, market_col])),
                "bias": float(np.nanmean(err[valid])),
                "mae": float(np.nanmean(np.abs(err[valid]))),
                "rmse": float(np.sqrt(np.nanmean(err[valid] ** 2))),
                "mape": float(np.nanmean(np.abs(rel[valid]))),
                "median_abs_error": float(np.nanmedian(np.abs(err[valid]))),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)


def grouped_error_summary(df: pd.DataFrame, price_col: str, group_col: str, market_col: str = "market_price") -> pd.DataFrame:
    tmp = df.copy()
    tmp["error"] = tmp[price_col] - tmp[market_col]
    tmp["abs_error"] = tmp["error"].abs()
    tmp["rel_error"] = np.where(tmp[market_col] > 0, tmp["error"] / tmp[market_col], np.nan)
    return (
        tmp.groupby(group_col, observed=True)
        .agg(
            n=("error", "count"),
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.nanmean(np.asarray(x) ** 2))),
            mape=("rel_error", lambda x: np.nanmean(np.abs(x))),
        )
        .reset_index()
    )


def crr_eqp_difference_summary(df: pd.DataFrame, vol_cols: list[str]) -> pd.DataFrame:
    rows = []
    for vol in vol_cols:
        crr = f"crr_{vol}_price"
        eqp = f"eqp_{vol}_price"
        if crr not in df or eqp not in df:
            continue
        diff = df[crr] - df[eqp]
        rows.append(
            {
                "volatility_input": vol,
                "n": int(diff.notna().sum()),
                "mean_crr_minus_eqp": float(diff.mean()),
                "mean_abs_difference": float(diff.abs().mean()),
                "max_abs_difference": float(diff.abs().max()),
            }
        )
    return pd.DataFrame(rows)
