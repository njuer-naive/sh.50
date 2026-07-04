import pandas as pd
from .config import OPTION_FILE, ETF_FILE, OPTION_NAME_FILTER


def _parse_trade_date(df: pd.DataFrame, col: str = "trade_date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d")
    return df


def load_option_data(path=OPTION_FILE, option_name_filter: str | None = OPTION_NAME_FILTER) -> pd.DataFrame:
    """
    Load option data and standardize date columns.

    The uploaded option file may contain contracts on several ETF underlyings.
    Because the supplied underlying file is 510050.SH, by default we keep only
    contracts whose name contains 华夏上证50ETF期权. Set option_name_filter=None
    if you intentionally want to keep every contract.
    """
    df = pd.read_csv(path)
    if option_name_filter and "name" in df.columns:
        before = len(df)
        df = df[df["name"].astype(str).str.contains(option_name_filter, na=False)].copy()
        print(f"Filtered option rows by {option_name_filter}: {before} -> {len(df)}")
    for col in ["trade_date", "list_date", "delist_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d")
    return df


def load_etf_data(path=ETF_FILE) -> pd.DataFrame:
    """Load 510050 ETF daily price data."""
    df = pd.read_csv(path)
    df = _parse_trade_date(df, "trade_date")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df
