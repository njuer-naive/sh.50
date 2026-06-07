"""
合并日线行情与合约基础信息
以 50ETF_option_daily_2022_now.csv 为基准, 左连接合约信息
输出: 50ETF_option_full.csv
"""

import pandas as pd

DAILY_CSV = '50ETF_option_daily_2022_now.csv'
CONTRACTS_CSV = '50ETF_option_contracts.csv'
OUTPUT = '50ETF_option_full.csv'

daily = pd.read_csv(DAILY_CSV, dtype={'trade_date': str, 'ts_code': str})
contracts = pd.read_csv(CONTRACTS_CSV, dtype={'ts_code': str})

merged = daily.merge(contracts, on='ts_code', how='left')

# 列顺序: 基本信息放前面, 行情放后面
cols = ['ts_code', 'name', 'call_put', 'exercise_price',
        'list_date', 'delist_date',
        'trade_date', 'open', 'high', 'low', 'close', 'settle',
        'vol', 'amount', 'oi']
merged = merged[cols]

merged.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

print(f"日线行数: {len(daily)}")
print(f"合约数: {contracts['ts_code'].nunique()}")
print(f"合并后行数: {len(merged)}")
print(f"匹配成功: {merged['name'].notna().sum()} / 未匹配: {merged['name'].isna().sum()}")
print(f"输出: {OUTPUT}")
