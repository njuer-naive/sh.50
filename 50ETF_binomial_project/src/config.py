from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "reports"

OPTION_FILE = RAW_DIR / "50ETF_option_full_with_rf.csv"
ETF_FILE = RAW_DIR / "510050_daily.csv"

TRADING_DAYS = 252
DAYS_IN_YEAR = 365
DEFAULT_TREE_STEPS = 50
DEFAULT_MARKET_PRICE_COL = "settle"
DIVIDEND_YIELD = 0.0

HIST_VOL_WINDOWS = [5, 30]
GARCH_ORDERS = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Only options linked to 510050.SH should be priced with the 510050 ETF price series.
OPTION_NAME_FILTER = "华夏上证50ETF期权"
