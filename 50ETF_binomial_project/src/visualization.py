from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .binomial import build_example_tree


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_volatility(vol_df: pd.DataFrame, vol_cols: list[str], out_path: Path):
    plt.figure(figsize=(12, 6))
    for col in vol_cols:
        plt.plot(vol_df["trade_date"], vol_df[col], label=col, linewidth=1.2)
    plt.title("510050 annualized volatility estimates")
    plt.xlabel("Trade date")
    plt.ylabel("Annualized volatility")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    _savefig(out_path)


def plot_error_summary(summary: pd.DataFrame, out_path: Path, top_n: int = 20):
    data = summary.sort_values("rmse").head(top_n).copy()
    plt.figure(figsize=(10, max(5, 0.35 * len(data))))
    plt.barh(data["model"], data["rmse"])
    plt.gca().invert_yaxis()
    plt.title("Pricing RMSE by tree model and volatility input")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.grid(axis="x", alpha=0.3)
    _savefig(out_path)


def plot_market_vs_model(df: pd.DataFrame, price_col: str, out_path: Path, max_points: int = 8000):
    plot_df = df[["market_price", price_col]].dropna()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)
    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df["market_price"], plot_df[price_col], s=5, alpha=0.35)
    max_val = float(np.nanmax([plot_df["market_price"].max(), plot_df[price_col].max()]))
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
    plt.title(f"Market price vs {price_col}")
    plt.xlabel("Market option price")
    plt.ylabel("Model option price")
    plt.grid(alpha=0.3)
    _savefig(out_path)


def plot_group_mae(group_summary: pd.DataFrame, group_col: str, out_path: Path):
    data = group_summary.copy()
    plt.figure(figsize=(8, 5))
    plt.bar(data[group_col].astype(str), data["mae"])
    plt.title(f"MAE by {group_col}")
    plt.xlabel(group_col)
    plt.ylabel("MAE")
    plt.grid(axis="y", alpha=0.3)
    _savefig(out_path)


def plot_crr_eqp_diff(diff_summary: pd.DataFrame, out_path: Path):
    data = diff_summary.sort_values("mean_abs_difference")
    plt.figure(figsize=(9, 5))
    plt.barh(data["volatility_input"], data["mean_abs_difference"])
    plt.title("Average absolute CRR-EQP price difference")
    plt.xlabel("Mean absolute difference")
    plt.ylabel("Volatility input")
    plt.grid(axis="x", alpha=0.3)
    _savefig(out_path)


def plot_tree_comparison(S: float, T: float, r: float, sigma: float, out_path: Path, steps: int = 4):
    crr = build_example_tree(S, T, r, sigma, steps=steps, tree_type="crr")
    eqp = build_example_tree(S, T, r, sigma, steps=steps, tree_type="eqp")
    plt.figure(figsize=(10, 5))
    for i, vals in enumerate(crr):
        plt.scatter([i] * len(vals), vals, marker="o", label="CRR" if i == 0 else None)
    for i, vals in enumerate(eqp):
        plt.scatter([i + 0.08] * len(vals), vals, marker="x", label="EQP" if i == 0 else None)
    plt.title("CRR tree vs EQP tree node prices, example option")
    plt.xlabel("Step")
    plt.ylabel("Underlying node price")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(out_path)
