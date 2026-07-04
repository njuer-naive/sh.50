import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    add_moneyness_and_groups,
    add_pricing_errors,
    crr_eqp_difference_summary,
    grouped_error_summary,
    summarize_errors,
)
from src.binomial import price_binomial_vectorized
from src.config import (
    DEFAULT_MARKET_PRICE_COL,
    DEFAULT_TREE_STEPS,
    DIVIDEND_YIELD,
    PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
    DAYS_IN_YEAR,
)
from src.data_loader import load_etf_data, load_option_data
from src.volatility import compute_all_volatility
from src.visualization import (
    plot_crr_eqp_diff,
    plot_error_summary,
    plot_group_mae,
    plot_market_vs_model,
    plot_tree_comparison,
    plot_volatility,
)


def prepare_dataset(option_df: pd.DataFrame, vol_df: pd.DataFrame, market_price_col: str) -> tuple[pd.DataFrame, list[str]]:
    vol_cols = [c for c in vol_df.columns if c.startswith("hist_vol_") or (c.startswith("garch_") and c.endswith("_vol"))]
    keep_cols = ["trade_date", "close", "log_return"] + vol_cols
    merged = option_df.merge(vol_df[keep_cols], on="trade_date", how="left", suffixes=("", "_underlying"))
    merged = merged.rename(columns={"close_underlying": "underlying_close"})
    # If the option file already has close, pandas suffixing creates close_underlying; if not, keep the ETF close name.
    if "underlying_close" not in merged.columns and "close" in vol_df.columns:
        merged = merged.rename(columns={"close": "underlying_close"})
    merged["market_price"] = merged[market_price_col].astype(float)
    merged["T"] = merged["days_to_maturity"].astype(float) / DAYS_IN_YEAR
    merged["r"] = merged["rf_rate_decimal"].astype(float)
    merged = add_moneyness_and_groups(merged)
    return merged, vol_cols


def run_pricing(df: pd.DataFrame, vol_cols: list[str], steps: int, dividend_yield: float) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    price_cols = []
    for vol_col in vol_cols:
        for tree_type in ["crr", "eqp"]:
            price_col = f"{tree_type}_{vol_col}_price"
            print(f"Pricing {price_col} ...")
            out[price_col] = price_binomial_vectorized(
                S=out["underlying_close"].values,
                K=out["exercise_price"].values,
                T=out["T"].values,
                r=out["r"].values,
                sigma=out[vol_col].values,
                option_type=out["call_put"].values,
                steps=steps,
                tree_type=tree_type,
                q=dividend_yield,
            )
            price_cols.append(price_col)
    return out, price_cols


def write_report(summary: pd.DataFrame, diff_summary: pd.DataFrame, steps: int, report_path: Path):
    best = summary.iloc[0]
    lines = [
        "# 50ETF Option Binomial Pricing Project Report",
        "",
        f"Tree steps used in the run: {steps}",
        "",
        "## Main outputs",
        "- data/processed/510050_volatility.csv: 5-day, 30-day historical volatility and GARCH conditional volatilities.",
        "- outputs/tables/option_pricing_results.csv: option rows with CRR/EQP prices under all volatility inputs.",
        "- outputs/tables/error_summary.csv: aggregate error metrics.",
        "- outputs/figures/: volatility, pricing, error and CRR-EQP comparison charts.",
        "",
        "## Best model by RMSE in this run",
        f"- Model: {best['model']}",
        f"- RMSE: {best['rmse']:.6f}",
        f"- MAE: {best['mae']:.6f}",
        f"- Bias: {best['bias']:.6f}",
        "",
        "## CRR vs EQP model difference",
        "CRR fixes u=exp(sigma*sqrt(dt)), d=1/u and adjusts the risk-neutral probability p. EQP/Jarrow-Rudd fixes p=0.5 and embeds the risk-neutral drift into u and d.",
        "",
        diff_summary.to_markdown(index=False),
        "",
        "## Notes",
        "The GARCH implementation is a self-contained Gaussian quasi-MLE implementation using scipy, so the project does not depend on the external arch package.",
        "The baseline dividend yield q is set to 0.0 in config.py. If ETF dividend data are available, replace q with a dividend yield estimate or use dividend-adjusted underlying prices.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="50ETF option pricing with CRR and EQP binomial trees.")
    parser.add_argument("--steps", type=int, default=DEFAULT_TREE_STEPS, help="Number of binomial tree steps.")
    parser.add_argument("--market-price-col", type=str, default=DEFAULT_MARKET_PRICE_COL, help="Option market price column, e.g. settle or close.")
    parser.add_argument("--dividend-yield", type=float, default=DIVIDEND_YIELD, help="Continuous dividend yield q.")
    args = parser.parse_args()

    for d in [PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    option_df = load_option_data()
    etf_df = load_etf_data()

    print("Computing volatility...")
    vol_df, garch_fit = compute_all_volatility(etf_df)
    vol_path = PROCESSED_DIR / "510050_volatility.csv"
    garch_fit_path = TABLE_DIR / "garch_fit_summary.csv"
    vol_df.to_csv(vol_path, index=False, encoding="utf-8-sig")
    garch_fit.to_csv(garch_fit_path, index=False, encoding="utf-8-sig")

    print("Merging option and volatility data...")
    dataset, vol_cols = prepare_dataset(option_df, vol_df, args.market_price_col)
    dataset.to_csv(PROCESSED_DIR / "option_dataset_with_volatility.csv", index=False, encoding="utf-8-sig")

    print("Running binomial pricing...")
    priced, price_cols = run_pricing(dataset, vol_cols, args.steps, args.dividend_yield)
    priced = add_pricing_errors(priced, price_cols)

    print("Writing tables...")
    pricing_path = TABLE_DIR / "option_pricing_results.csv"
    priced.to_csv(pricing_path, index=False, encoding="utf-8-sig")
    summary = summarize_errors(priced, price_cols)
    summary.to_csv(TABLE_DIR / "error_summary.csv", index=False, encoding="utf-8-sig")

    baseline_col = "crr_hist_vol_30d_price" if "crr_hist_vol_30d_price" in price_cols else price_cols[0]
    grouped_tables = {
        "call_put": grouped_error_summary(priced, baseline_col, "call_put"),
        "moneyness_group": grouped_error_summary(priced, baseline_col, "moneyness_group"),
        "maturity_group": grouped_error_summary(priced, baseline_col, "maturity_group"),
        "liquidity_group": grouped_error_summary(priced, baseline_col, "liquidity_group"),
    }
    for name, table in grouped_tables.items():
        table.to_csv(TABLE_DIR / f"group_summary_{name}.csv", index=False, encoding="utf-8-sig")

    diff_summary = crr_eqp_difference_summary(priced, vol_cols)
    diff_summary.to_csv(TABLE_DIR / "crr_eqp_difference_summary.csv", index=False, encoding="utf-8-sig")

    print("Drawing figures...")
    plot_volatility(vol_df, vol_cols, FIGURE_DIR / "volatility_series.png")
    plot_error_summary(summary, FIGURE_DIR / "rmse_by_model.png")
    if baseline_col in priced:
        plot_market_vs_model(priced, baseline_col, FIGURE_DIR / "market_vs_model_baseline_crr_hist30.png")
        for group_name, table in grouped_tables.items():
            plot_group_mae(table, group_name, FIGURE_DIR / f"mae_by_{group_name}.png")
    plot_crr_eqp_diff(diff_summary, FIGURE_DIR / "crr_eqp_mean_abs_difference.png")

    # Example tree: choose a relatively normal, valid row.
    valid_example = priced.dropna(subset=["underlying_close", "T", "r", "hist_vol_30d"])
    valid_example = valid_example[(valid_example["T"] > 0) & (valid_example["hist_vol_30d"] > 0)]
    if len(valid_example) > 0:
        ex = valid_example.iloc[len(valid_example) // 2]
        plot_tree_comparison(
            S=float(ex["underlying_close"]),
            T=float(ex["T"]),
            r=float(ex["r"]),
            sigma=float(ex["hist_vol_30d"]),
            out_path=FIGURE_DIR / "crr_vs_eqp_tree_example.png",
            steps=4,
        )

    write_report(summary, diff_summary, args.steps, REPORT_DIR / "summary_report.md")
    print("Done.")
    print(f"Volatility CSV: {vol_path}")
    print(f"Pricing results: {pricing_path}")
    print(f"Best model by RMSE: {summary.iloc[0]['model']}")


if __name__ == "__main__":
    main()
