"""
上证50ETF(510050)日线行情数据获取
时间范围: 2022-01-01 至今
输出: 510050_daily.csv
"""

import tushare as ts
import pandas as pd
from datetime import datetime

TOKEN = '80b324ca2fc11495a301563b81ac7a0d0507eec63008dd2c01a320f7'
START_DATE = '20220101'
END_DATE = datetime.now().strftime('%Y%m%d')
OUTPUT = '510050_daily.csv'

pro = ts.pro_api(TOKEN)

# 510050 是ETF基金，用fund_daily接口
df = pro.fund_daily(ts_code='510050.SH', start_date=START_DATE, end_date=END_DATE,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount')

df = df.sort_values('trade_date')
df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

print(f"下载完成: {len(df)} 行")
print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"输出: {OUTPUT}")
print(df.head())
