from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm

try:
    from arch import arch_model
except ImportError:  # pragma: no cover
    arch_model = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OPTION_FILE = DATA_DIR / "50ETF_option_full_with_rf.csv"
ETF_FILE = DATA_DIR / "510050_daily.csv"

HISTORICAL_WINDOWS = [3, 5, 10, 30, 60]
GARCH_SPECS = [(1, 1), (1, 2), (2, 1), (2, 2)]
TRADING_DAYS = 252
CALENDAR_DAYS = 365
MAX_IV = 5.0

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.35,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    for attempt in range(20):
        try:
            tmp_path.replace(out_path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def bsm_price(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tau: np.ndarray | float,
    rate: np.ndarray | float,
    sigma: np.ndarray | float,
    option_type: np.ndarray | str,
) -> np.ndarray | float:
    spot_arr = np.asarray(spot, dtype=float)
    strike_arr = np.asarray(strike, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    opt_arr = np.asarray(option_type)

    out = np.full(np.broadcast(spot_arr, strike_arr, tau_arr, rate_arr, sigma_arr).shape, np.nan)
    spot_b, strike_b, tau_b, rate_b, sigma_b = np.broadcast_arrays(
        spot_arr, strike_arr, tau_arr, rate_arr, sigma_arr
    )
    opt_b = np.broadcast_to(opt_arr, out.shape)
    valid = (spot_b > 0) & (strike_b > 0) & (tau_b > 0) & (sigma_b > 0)
    if not np.any(valid):
        return out.item() if out.ndim == 0 else out

    d1 = (
        np.log(spot_b[valid] / strike_b[valid])
        + (rate_b[valid] + 0.5 * sigma_b[valid] ** 2) * tau_b[valid]
    ) / (sigma_b[valid] * np.sqrt(tau_b[valid]))
    d2 = d1 - sigma_b[valid] * np.sqrt(tau_b[valid])
    disc_strike = strike_b[valid] * np.exp(-rate_b[valid] * tau_b[valid])

    call_price = spot_b[valid] * norm.cdf(d1) - disc_strike * norm.cdf(d2)
    put_price = disc_strike * norm.cdf(-d2) - spot_b[valid] * norm.cdf(-d1)
    out[valid] = np.where(opt_b[valid] == "C", call_price, put_price)
    return out.item() if out.ndim == 0 else out


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    options = pd.read_csv(OPTION_FILE)
    etf = pd.read_csv(ETF_FILE)

    options["trade_date"] = pd.to_datetime(options["trade_date"].astype(str), format="%Y%m%d")
    options["list_date"] = pd.to_datetime(options["list_date"].astype(str), format="%Y%m%d")
    options["delist_date"] = pd.to_datetime(options["delist_date"].astype(str), format="%Y%m%d")
    etf["trade_date"] = pd.to_datetime(etf["trade_date"].astype(str), format="%Y%m%d")

    etf = etf.sort_values("trade_date").copy()
    etf["log_return"] = np.log(etf["close"] / etf["close"].shift(1))
    etf["trading_horizon"] = np.arange(len(etf), dtype=int)

    options["market_price"] = options["close"].where(options["close"].notna(), options["settle"])
    options["tau"] = options["days_to_maturity"] / CALENDAR_DAYS
    merged = options.merge(
        etf[["trade_date", "close", "log_return", "trading_horizon"]].rename(
            columns={"close": "underlying_close"}
        ),
        on="trade_date",
        how="left",
    )
    merged = merged[
        (merged["market_price"] > 0)
        & (merged["underlying_close"] > 0)
        & (merged["exercise_price"] > 0)
        & (merged["tau"] > 0)
        & merged["call_put"].isin(["C", "P"])
    ].copy()
    merged["moneyness"] = merged["underlying_close"] / merged["exercise_price"]
    merged["forecast_horizon_days"] = np.ceil(merged["days_to_maturity"] * TRADING_DAYS / CALENDAR_DAYS).clip(
        lower=1
    )
    merged["forecast_horizon_days"] = merged["forecast_horizon_days"].astype(int)

    call = merged["call_put"].eq("C")
    discounted_strike = merged["exercise_price"] * np.exp(-merged["rf_rate_decimal"] * merged["tau"])
    merged["bsm_lower_bound"] = np.where(
        call,
        np.maximum(merged["underlying_close"] - discounted_strike, 0.0),
        np.maximum(discounted_strike - merged["underlying_close"], 0.0),
    )
    merged["bsm_upper_bound"] = np.where(call, merged["underlying_close"], discounted_strike)
    valid_bounds = (
        (merged["market_price"] >= merged["bsm_lower_bound"] - 1e-6)
        & (merged["market_price"] <= merged["bsm_upper_bound"] + 1e-6)
    )

    excluded = merged.loc[~valid_bounds].copy()
    if not excluded.empty:
        excluded_cols = [
            "ts_code",
            "name",
            "call_put",
            "trade_date",
            "exercise_price",
            "days_to_maturity",
            "underlying_close",
            "market_price",
            "rf_rate_decimal",
            "bsm_lower_bound",
            "bsm_upper_bound",
            "moneyness",
        ]
        write_csv(excluded[excluded_cols], BASE_DIR / "excluded_by_bsm_bounds.csv")

    return merged.loc[valid_bounds].copy(), etf


def implied_volatility_vectorized(options: pd.DataFrame) -> np.ndarray:
    price = options["market_price"].to_numpy(dtype=float)
    spot = options["underlying_close"].to_numpy(dtype=float)
    strike = options["exercise_price"].to_numpy(dtype=float)
    tau = options["tau"].to_numpy(dtype=float)
    rate = options["rf_rate_decimal"].to_numpy(dtype=float)
    option_type = options["call_put"].to_numpy()

    call = option_type == "C"
    discounted_strike = strike * np.exp(-rate * tau)
    lower_bound = np.where(call, np.maximum(spot - discounted_strike, 0.0), np.maximum(discounted_strike - spot, 0.0))
    upper_bound = np.where(call, spot, discounted_strike)
    valid = (
        np.isfinite(price)
        & np.isfinite(spot)
        & np.isfinite(strike)
        & np.isfinite(tau)
        & np.isfinite(rate)
        & (price > 0)
        & (spot > 0)
        & (strike > 0)
        & (tau > 0)
        & (price >= lower_bound - 1e-8)
        & (price <= upper_bound + 1e-8)
    )

    iv = np.full(len(options), np.nan)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return iv

    low = np.full(len(idx), 1e-6)
    high = np.full(len(idx), MAX_IV)
    p = price[idx]
    s = spot[idx]
    k = strike[idx]
    t = tau[idx]
    r = rate[idx]
    typ = option_type[idx]

    bracketed = (bsm_price(s, k, t, r, low, typ) - p <= 1e-8) & (
        bsm_price(s, k, t, r, high, typ) - p >= -1e-8
    )
    active_idx = idx[bracketed]
    if len(active_idx) == 0:
        return iv

    low = low[bracketed]
    high = high[bracketed]
    p = p[bracketed]
    s = s[bracketed]
    k = k[bracketed]
    t = t[bracketed]
    r = r[bracketed]
    typ = typ[bracketed]

    for _ in range(60):
        mid = (low + high) / 2.0
        model = bsm_price(s, k, t, r, mid, typ)
        too_high = model >= p
        high = np.where(too_high, mid, high)
        low = np.where(too_high, low, mid)

    solved = (low + high) / 2.0
    residual = np.abs(bsm_price(s, k, t, r, solved, typ) - p)
    iv[active_idx] = np.where(residual <= 1e-5, solved, np.nan)
    return iv


def add_implied_volatility(options: pd.DataFrame) -> pd.DataFrame:
    cache_file = BASE_DIR / "market_implied_volatility.csv"
    if cache_file.exists():
        iv = pd.read_csv(cache_file)
        iv["trade_date"] = pd.to_datetime(iv["trade_date"])
        keep = ["ts_code", "trade_date", "implied_vol"]
        return options.merge(iv[keep], on=["ts_code", "trade_date"], how="left")

    options = options.copy()
    options["implied_vol"] = implied_volatility_vectorized(options)
    iv_cols = [
        "ts_code",
        "name",
        "call_put",
        "trade_date",
        "exercise_price",
        "days_to_maturity",
        "underlying_close",
        "market_price",
        "rf_rate_decimal",
        "moneyness",
        "implied_vol",
    ]
    write_csv(
        options[iv_cols],
        cache_file,
    )
    return options


def estimate_historical_volatility(etf: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = etf[["trade_date", "close", "log_return"]].copy()
    vol["sample_mean_daily"] = vol["log_return"].rolling(window=window, min_periods=window).mean()
    vol["sample_std_daily"] = vol["log_return"].rolling(window=window, min_periods=window).std(ddof=1)
    vol["sample_mean_annualized"] = vol["sample_mean_daily"] * TRADING_DAYS
    vol["model_vol"] = vol["sample_std_daily"] * np.sqrt(TRADING_DAYS)
    vol["model_name"] = f"historical_{window}d"
    return vol


def average_garch_variance(h1: np.ndarray, omega: float, persistence: float, horizon: np.ndarray) -> np.ndarray:
    horizon = np.asarray(horizon, dtype=int).clip(min=1)
    if persistence >= 0.999:
        return h1
    long_run = omega / max(1.0 - persistence, 1e-12)
    powers = persistence ** np.arange(horizon.max())
    prefix = np.cumsum(powers)
    mean_decay = prefix[horizon - 1] / horizon
    return long_run + (h1 - long_run) * mean_decay


def estimate_garch_volatility(
    etf: pd.DataFrame,
    options: pd.DataFrame,
    p: int = 1,
    q: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if arch_model is None:
        raise RuntimeError("The 'arch' package is required for GARCH estimation.")

    model_name = f"garch_{p}_{q}"
    model_label = f"GARCH({p},{q})"
    return_data = etf.dropna(subset=["log_return"])[["trade_date", "close", "log_return"]].copy()
    returns = return_data.set_index("trade_date")["log_return"] * 100
    model = arch_model(returns, mean="Constant", vol="GARCH", p=p, q=q, dist="normal", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    params = result.params
    max_horizon = int(options["forecast_horizon_days"].max())
    forecast_variance = result.forecast(horizon=max_horizon, start=0, reindex=True).variance
    forecast_variance = forecast_variance.reindex(returns.index)

    base = return_data.copy()
    base["garch_h1_variance_pct2"] = forecast_variance.iloc[:, 0].to_numpy(dtype=float)
    base["model_vol"] = np.sqrt((base["garch_h1_variance_pct2"] / 10000) * TRADING_DAYS)
    base["model_name"] = model_name

    variance_matrix = forecast_variance.to_numpy(dtype=float)
    cumulative_variance = np.cumsum(variance_matrix, axis=1)
    date_to_position = {date: i for i, date in enumerate(forecast_variance.index)}
    positions = options["trade_date"].map(date_to_position)
    horizons = options["forecast_horizon_days"].clip(upper=max_horizon).to_numpy(dtype=int)
    avg_pct2 = np.full(len(options), np.nan)
    valid_pos = positions.notna().to_numpy()
    pos = positions.loc[valid_pos].to_numpy(dtype=int)
    h = horizons[valid_pos]
    avg_pct2[valid_pos] = cumulative_variance[pos, h - 1] / h
    option_vol = np.sqrt((avg_pct2 / 10000) * TRADING_DAYS)

    alpha_sum = sum(float(params.get(f"alpha[{i}]", 0.0)) for i in range(1, p + 1))
    beta_sum = sum(float(params.get(f"beta[{j}]", 0.0)) for j in range(1, q + 1))
    row = {
        "model": model_label,
        "p": p,
        "q": q,
        "mu_pct": float(params.get("mu", np.nan)),
        "omega": float(params.get("omega", np.nan)),
        "alpha_sum": alpha_sum,
        "beta_sum": beta_sum,
        "persistence": alpha_sum + beta_sum,
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }
    for i in range(1, p + 1):
        row[f"alpha_{i}"] = float(params.get(f"alpha[{i}]", np.nan))
    for j in range(1, q + 1):
        row[f"beta_{j}"] = float(params.get(f"beta[{j}]", np.nan))

    params_df = pd.DataFrame([row])
    option_vol_df = options[["ts_code", "trade_date", "forecast_horizon_days"]].copy()
    option_vol_df["model_vol"] = option_vol
    return base, option_vol_df, params_df


def estimate_stochastic_volatility(etf: pd.DataFrame, options: pd.DataFrame, window: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_var = etf["log_return"].rolling(window=window, min_periods=window).var(ddof=1)
    rv = daily_var.dropna().clip(lower=1e-10)
    x = rv.shift(1).dropna()
    y = rv.loc[x.index]
    slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    phi = float(np.clip(slope, 0.0, 0.999))
    theta = float(max(intercept / max(1.0 - phi, 1e-8), rv.mean()))

    vol = etf[["trade_date", "close", "log_return"]].copy()
    vol["sv_current_variance"] = daily_var
    vol["model_vol"] = np.sqrt(vol["sv_current_variance"] * TRADING_DAYS)
    vol["model_name"] = "stochastic_volatility"

    lookup = vol[["trade_date", "sv_current_variance"]].copy()
    priced_vol = options[["ts_code", "trade_date", "forecast_horizon_days"]].merge(lookup, on="trade_date", how="left")
    h = priced_vol["forecast_horizon_days"].to_numpy(dtype=int).clip(min=1)
    current_var = priced_vol["sv_current_variance"].to_numpy(dtype=float)
    if phi <= 1e-10:
        avg_var = np.full_like(current_var, theta)
    else:
        decay_mean = phi * (1.0 - phi**h) / (h * (1.0 - phi))
        avg_var = theta + decay_mean * (current_var - theta)
    priced_vol["model_vol"] = np.sqrt(np.clip(avg_var, 1e-12, None) * TRADING_DAYS)

    innovation = y - (intercept + slope * x)
    params_df = pd.DataFrame(
        [
            {
                "model": "Heston-style stochastic volatility",
                "realized_variance_window": window,
                "daily_theta": theta,
                "daily_phi": phi,
                "annualized_long_run_vol": math.sqrt(theta * TRADING_DAYS),
                "variance_innovation_std": float(innovation.std(ddof=1)),
            }
        ]
    )
    return vol, priced_vol[["ts_code", "trade_date", "forecast_horizon_days", "model_vol"]], params_df


def error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    err = df["pricing_error"]
    abs_err = err.abs()
    pct = df["relative_error"].abs().replace([np.inf, -np.inf], np.nan)
    smape = (2 * abs_err / (df["bsm_price"].abs() + df["market_price"].abs())).replace([np.inf, -np.inf], np.nan)
    floored_pct = (abs_err / df["market_price"].clip(lower=0.01)).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        [
            {
                "n": int(df.shape[0]),
                "mean_error": err.mean(),
                "mae": abs_err.mean(),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "median_abs_error": abs_err.median(),
                "mape": pct.mean(),
                "smape": smape.mean(),
                "mape_price_floor_0_01": floored_pct.mean(),
                "bias_ratio": (err / df["market_price"]).replace([np.inf, -np.inf], np.nan).mean(),
                "correlation_market_model": df[["market_price", "bsm_price"]].corr().iloc[0, 1],
            }
        ]
    )


def grouped_error_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "mae": g["pricing_error"].abs().mean(),
                    "rmse": np.sqrt(np.mean(g["pricing_error"] ** 2)),
                    "mape": g["relative_error"].abs().replace([np.inf, -np.inf], np.nan).mean(),
                    "smape": (
                        2 * g["pricing_error"].abs() / (g["bsm_price"].abs() + g["market_price"].abs())
                    ).replace([np.inf, -np.inf], np.nan).mean(),
                    "mean_error": g["pricing_error"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def make_surface_data(priced: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plot_df = priced[
        priced[value_col].between(0, 3)
        & priced["moneyness"].between(0.75, 1.25)
        & priced["trade_date"].notna()
    ].copy()
    plot_df["month"] = plot_df["trade_date"].dt.to_period("M").dt.to_timestamp()
    plot_df["moneyness_bin"] = pd.cut(plot_df["moneyness"], bins=np.linspace(0.75, 1.25, 26))
    surface = (
        plot_df.groupby(["month", "moneyness_bin"], observed=True)
        .agg(date=("month", "first"), moneyness=("moneyness", "mean"), value=(value_col, "mean"))
        .dropna()
        .reset_index(drop=True)
    )
    pivot = surface.pivot(index="moneyness", columns="date", values="value").sort_index()
    pivot = pivot.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
    x_dates = mdates.date2num(pd.to_datetime(pivot.columns))
    y_money = pivot.index.to_numpy(dtype=float)
    x_grid, y_grid = np.meshgrid(x_dates, y_money)
    z_grid = pivot.to_numpy(dtype=float)
    return x_grid, y_grid, z_grid


def save_volatility_surface(priced: pd.DataFrame, out_dir: Path, label: str) -> None:
    if priced["implied_vol"].dropna().empty or priced["model_vol"].dropna().empty:
        return

    market_x, market_y, market_z = make_surface_data(priced, "implied_vol")
    model_x, model_y, model_z = make_surface_data(priced, "model_vol")
    fig = plt.figure(figsize=(12.5, 5.2))
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
    panels = [
        (axes[0], market_x, market_y, market_z, "Market implied volatility"),
        (axes[1], model_x, model_y, model_z, f"{label} volatility"),
    ]
    for ax, x_grid, y_grid, z_grid, title in panels:
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.96)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Moneyness S/K")
        ax.set_zlabel("Annualized volatility")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
        ax.view_init(elev=26, azim=-132)
        ax.grid(False)
    save_figure(fig, out_dir / "volatility_surface_3d")


def save_plots(priced: pd.DataFrame, daily_vol: pd.DataFrame, out_dir: Path, label: str) -> None:
    apply_publication_style()

    daily_iv = (
        priced.loc[priced["implied_vol"].between(0, 3), ["trade_date", "implied_vol"]]
        .groupby("trade_date")
        .mean()
        .reset_index()
    )
    daily_model = daily_vol[["trade_date", "model_vol"]].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(daily_iv["trade_date"], daily_iv["implied_vol"], label="Market IV", color=OKABE_ITO["blue"])
    ax.plot(daily_model["trade_date"], daily_model["model_vol"], label=label, color=OKABE_ITO["orange"])
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Annualized volatility")
    ax.grid(True)
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "volatility_time_series")

    plot_df = priced.dropna(subset=["implied_vol", "model_vol"]).copy()
    usable_dates = plot_df.loc[plot_df["implied_vol"].between(0, 3), "trade_date"]
    if not usable_dates.empty:
        selected_date = usable_dates.max()
        smile = plot_df[
            (plot_df["trade_date"] == selected_date)
            & plot_df["implied_vol"].between(0, 3)
            & plot_df["moneyness"].between(0.75, 1.25)
        ].copy()
        if not smile.empty:
            smile["moneyness_bin"] = pd.cut(smile["moneyness"], bins=np.linspace(0.75, 1.25, 21))
            curve = (
                smile.groupby("moneyness_bin", observed=True)
                .agg(moneyness=("moneyness", "mean"), implied_vol=("implied_vol", "mean"), model_vol=("model_vol", "mean"))
                .dropna()
                .reset_index(drop=True)
            )
            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(curve["moneyness"], curve["implied_vol"], marker="o", ms=3.5, label="Market IV", color=OKABE_ITO["blue"])
            ax.plot(curve["moneyness"], curve["model_vol"], marker="s", ms=3.2, label=label, color=OKABE_ITO["red"])
            ax.set_xlabel("Moneyness S/K")
            ax.set_ylabel("Annualized volatility")
            ax.grid(True)
            ax.legend(loc="best")
            save_figure(fig, out_dir / "implied_vol_smile")

    sample = priced.sample(min(20000, len(priced)), random_state=42)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(sample["market_price"], sample["bsm_price"], s=6, alpha=0.23, color=OKABE_ITO["blue"], linewidths=0)
    lim = float(np.nanmax([sample["market_price"].max(), sample["bsm_price"].max()]))
    ax.plot([0, lim], [0, lim], color=OKABE_ITO["red"], lw=1.0, label="45-degree line")
    ax.set_xlabel("Market option price")
    ax.set_ylabel("Model price")
    ax.grid(True)
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "market_vs_model_price")

    priced = priced.copy()
    priced["moneyness_bucket"] = pd.cut(
        priced["moneyness"],
        bins=[0, 0.9, 0.97, 1.03, 1.1, np.inf],
        labels=["S/K<=0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", "S/K>1.10"],
    )
    bucket = grouped_error_metrics(priced.dropna(subset=["moneyness_bucket"]), ["moneyness_bucket"])
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.bar(bucket["moneyness_bucket"].astype(str), bucket["mae"], color=OKABE_ITO["green"], width=0.72)
    ax.set_xlabel("Moneyness bucket")
    ax.set_ylabel("MAE")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, out_dir / "mae_by_moneyness")
    save_volatility_surface(priced, out_dir, label)


def run_pricing_model(
    options: pd.DataFrame,
    daily_vol: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    model_label: str,
    option_vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(daily_vol, out_dir / "volatility_estimates.csv")

    if option_vol is None:
        priced = options.merge(daily_vol[["trade_date", "model_vol"]], on="trade_date", how="left")
    else:
        priced = options.merge(
            option_vol[["ts_code", "trade_date", "forecast_horizon_days", "model_vol"]],
            on=["ts_code", "trade_date", "forecast_horizon_days"],
            how="left",
        )

    priced = priced[priced["model_vol"].notna() & (priced["model_vol"] > 0)].copy()
    priced["bsm_price"] = bsm_price(
        priced["underlying_close"].to_numpy(),
        priced["exercise_price"].to_numpy(),
        priced["tau"].to_numpy(),
        priced["rf_rate_decimal"].to_numpy(),
        priced["model_vol"].to_numpy(),
        priced["call_put"].to_numpy(),
    )
    priced = priced[priced["bsm_price"].notna()].copy()
    priced["pricing_error"] = priced["bsm_price"] - priced["market_price"]
    priced["absolute_error"] = priced["pricing_error"].abs()
    priced["relative_error"] = priced["pricing_error"] / priced["market_price"]

    columns = [
        "ts_code",
        "name",
        "call_put",
        "trade_date",
        "exercise_price",
        "days_to_maturity",
        "forecast_horizon_days",
        "rf_rate_decimal",
        "underlying_close",
        "market_price",
        "bsm_price",
        "pricing_error",
        "absolute_error",
        "relative_error",
        "moneyness",
        "implied_vol",
        "model_vol",
    ]
    write_csv(priced[columns], out_dir / "pricing_results.csv")

    metrics = error_metrics(priced)
    metrics.insert(0, "model", model_name)
    write_csv(metrics, out_dir / "error_metrics.csv")

    write_csv(grouped_error_metrics(priced, ["call_put"]), out_dir / "error_by_call_put.csv")
    priced["maturity_bucket"] = pd.cut(
        priced["days_to_maturity"],
        bins=[0, 7, 30, 90, np.inf],
        labels=["1-7d", "8-30d", "31-90d", "90d+"],
    )
    write_csv(
        grouped_error_metrics(priced.dropna(subset=["maturity_bucket"]), ["maturity_bucket"]),
        out_dir / "error_by_maturity.csv",
    )
    priced["moneyness_bucket"] = pd.cut(
        priced["moneyness"],
        bins=[0, 0.9, 0.97, 1.03, 1.1, np.inf],
        labels=["S/K<=0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", "S/K>1.10"],
    )
    write_csv(
        grouped_error_metrics(priced.dropna(subset=["moneyness_bucket"]), ["moneyness_bucket"]),
        out_dir / "error_by_moneyness.csv",
    )
    write_csv(grouped_error_metrics(priced, ["trade_date"]), out_dir / "daily_error_metrics.csv")
    save_plots(priced, daily_vol, out_dir, model_label)
    return metrics


def write_report(summary: pd.DataFrame, options: pd.DataFrame) -> None:
    valid_iv = int(options["implied_vol"].notna().sum())
    total = int(len(options))
    excluded_file = BASE_DIR / "excluded_by_bsm_bounds.csv"
    excluded_count = len(pd.read_csv(excluded_file)) if excluded_file.exists() else 0
    best_mae = summary.loc[summary["mae"].idxmin()]
    best_rmse = summary.loc[summary["rmse"].idxmin()]
    lines = [
        "# 上证 50ETF 期权 BSM 定价实验",
        "",
        "## 一、实验目标",
        "",
        "本实验使用 Black-Scholes-Merton（BSM）模型对上证 50ETF 期权进行定价，并比较多种波动率输入对定价误差的影响。实验首先使用 3、5、10、30、60 个交易日的历史样本标准差估计年化波动率，随后加入多组 GARCH(p,q) 条件异方差模型和 Heston 风格的随机波动率模型，形成可复现实验管线。",
        "",
        "## 二、数据与样本处理",
        "",
        "- `bsm/data/50ETF_option_full_with_rf.csv`：期权合约日行情、行权价、剩余到期天数和对应无风险利率。",
        "- `bsm/data/510050_daily.csv`：上证 50ETF 日行情，用于计算对数收益率和估计波动率。",
        f"- 经过价格、标的价格、行权价和剩余期限等基础筛选后，可用于普通 BSM 框架的样本数为 `{total}`。",
        f"- 因部分调整合约在缺少合约调整系数时会违反无股息 BSM 价格上下界，已将 `{excluded_count}` 条记录保存到 `excluded_by_bsm_bounds.csv`，不纳入误差比较。",
        f"- 从市场价格反解得到的隐含波动率有效样本数为 `{valid_iv}`。",
        "",
        "## 三、实验方法",
        "",
        "1. 市场价格使用期权 `close` 字段；若收盘价缺失，则使用 `settle` 字段作为替代。",
        "2. 到期时间使用 `days_to_maturity / 365`，无风险利率使用数据集中给出的 `rf_rate_decimal`。",
        "3. 历史波动率模型使用 ETF 对数收益率的滚动样本标准差，并按 `sqrt(252)` 年化；样本均值同步输出，用于说明估计窗口内收益率水平。",
        f"4. GARCH 模型使用 `arch` 包估计多组参数规格：{', '.join([f'GARCH({p},{q})' for p, q in GARCH_SPECS])}。对每个交易日，使用模型的多步条件方差预测，并根据期权剩余期限计算未来平均条件方差，再年化后代入 BSM 定价。",
        "5. 随机波动率模型使用 30 日实现方差构造离散均值回复方差过程，形式上对应 Heston 方差过程的离散近似；定价时采用未来平均方差的条件期望作为 BSM 等效波动率。",
        "6. 对所有模型输出 MAE、RMSE、Median AE、MAPE、SMAPE、带 0.01 价格下限的 MAPE、平均偏差和市场价格/模型价格相关系数。",
        "",
        "## 四、核心结果",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"- MAE 最低的模型为 `{best_mae['model']}`，MAE 为 `{best_mae['mae']:.6f}`。",
        f"- RMSE 最低的模型为 `{best_rmse['model']}`，RMSE 为 `{best_rmse['rmse']:.6f}`。",
        "- 原始 MAPE 对接近 0 的期权价格非常敏感，因此报告中同时给出 SMAPE 与设置 0.01 价格下限后的 MAPE。",
        "",
        "## 五、生成文件说明",
        "",
        "- `summary_metrics.csv`：所有波动率模型的总误差对比表。",
        "- `market_implied_volatility.csv`：由市场期权价格反解得到的隐含波动率缓存，后续复现实验时可直接复用。",
        "- `excluded_by_bsm_bounds.csv`：违反普通无股息 BSM 上下界、因缺少调整系数而未纳入比较的合约记录。",
        "- `window_*d/`：不同历史波动率窗口的结果目录。",
        "- `model_garch/`：基准 GARCH(1,1) 波动率预测与 BSM 定价结果。",
        "- `model_garch_*_*/`：其他 GARCH(p,q) 参数组合的波动率预测与 BSM 定价结果，例如 `model_garch_1_2/`、`model_garch_2_1/`、`model_garch_2_2/`。",
        "- `model_stochastic_volatility/`：随机波动率模型预测与 BSM 等效定价结果。",
        "- 各模型目录中的 `pricing_results.csv` 保存逐条期权定价结果；`error_metrics.csv` 保存总体误差；`error_by_call_put.csv`、`error_by_maturity.csv`、`error_by_moneyness.csv` 分别按期权类型、期限和价内外程度分组。",
        "- `volatility_time_series.(png|pdf)` 为市场隐含波动率与模型波动率的时间序列对比。",
        "- `implied_vol_smile.(png|pdf)` 为最近交易日的隐含波动率微笑与模型波动率对比。",
        "- `volatility_surface_3d.(png|pdf)` 为三维波动率曲面，横轴包含不同日期，纵轴为 moneyness，竖轴为年化波动率。",
        "- `market_vs_model_price.(png|pdf)` 为市场价格与模型价格散点图；`mae_by_moneyness.(png|pdf)` 为不同 moneyness 区间的 MAE 柱状图。",
        "",
    ]
    (BASE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    apply_publication_style()
    options, etf = load_and_prepare()
    options = add_implied_volatility(options)

    all_metrics: list[pd.DataFrame] = []
    for window in HISTORICAL_WINDOWS:
        daily_vol = estimate_historical_volatility(etf, window)
        metrics = run_pricing_model(
            options,
            daily_vol,
            BASE_DIR / f"window_{window}d",
            f"historical_{window}d",
            f"Historical {window}d",
        )
        all_metrics.append(metrics)

    for p, q in GARCH_SPECS:
        garch_daily, garch_option_vol, garch_params = estimate_garch_volatility(etf, options, p=p, q=q)
        garch_dir = BASE_DIR / ("model_garch" if (p, q) == (1, 1) else f"model_garch_{p}_{q}")
        garch_dir.mkdir(parents=True, exist_ok=True)
        write_csv(garch_params, garch_dir / "model_parameters.csv")
        all_metrics.append(
            run_pricing_model(
                options,
                garch_daily,
                garch_dir,
                f"garch_{p}_{q}",
                f"GARCH({p},{q})",
                garch_option_vol,
            )
        )

    sv_daily, sv_option_vol, sv_params = estimate_stochastic_volatility(etf, options)
    sv_dir = BASE_DIR / "model_stochastic_volatility"
    sv_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sv_params, sv_dir / "model_parameters.csv")
    all_metrics.append(
        run_pricing_model(
            options,
            sv_daily,
            sv_dir,
            "stochastic_volatility",
            "Stochastic volatility",
            sv_option_vol,
        )
    )

    summary = pd.concat(all_metrics, ignore_index=True).sort_values("mae")
    write_csv(summary, BASE_DIR / "summary_metrics.csv")
    write_report(summary, options)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
