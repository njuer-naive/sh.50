"""
上证50ETF期权日线行情数据获取
标的: 510050 | 交易所: SSE | 时间: 2022-01-01 至今
输出: 50ETF_option_daily_2022_now.csv (ts_code,trade_date,OHLCV,settle,oi等)
"""

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ==================== 配置 ====================
TOKEN = '80b324ca2fc11495a301563b81ac7a0d0507eec63008dd2c01a320f7'
START_DATE = '20220101'
END_DATE = datetime.now().strftime('%Y%m%d')
OUTPUT_CSV = '50ETF_option_daily_2022_now.csv'
LOG_FILE = 'fetch_log.txt'
SLEEP = 0.3  # 接口调用间隔(秒)，免费用户建议>=0.2s
CHUNK_DAYS = 5  # 每次拉取天数，避免超5000行限制
# =============================================

pro = ts.pro_api(TOKEN)


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)


def date_chunks(start, end, step_days):
    """生成 [start_str, end_str] 之间的日期区间, 步长step_days"""
    fmt = '%Y%m%d'
    d = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    while d <= e:
        nxt = min(d + timedelta(days=step_days - 1), e)
        yield d.strftime(fmt), nxt.strftime(fmt)
        d = nxt + timedelta(days=1)


def load_existing():
    """加载已有数据，返回已覆盖的(trade_date)集合用于断点续传"""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    existing = pd.read_csv(OUTPUT_CSV, dtype={'trade_date': str})
    dates = set(existing['trade_date'].unique())
    log(f"检测到已有数据: {len(existing)} 行, {len(dates)} 个交易日")
    return dates


def main():
    # 1. 获取50ETF期权合约列表 (用ts_code过滤，不用name)
    log("获取合约基础信息...")
    basic = pro.opt_basic(exchange='SSE', fields='ts_code,name,list_date,delist_date,exercise_price,call_put')
    # 510050是上证50ETF的代码，期权ts_code格式: 1000XXXX.SH
    # opt_basic的ts_code中包含标的代码信息，用name二次确认
    mask = basic['name'].str.contains('50ETF', na=False)
    contracts = basic[mask].copy()
    log(f"获取到 {len(contracts)} 个50ETF期权合约")

    # 保存合约列表，方便对照
    contracts.to_csv('50ETF_option_contracts.csv', index=False, encoding='utf-8-sig')
    log(f"合约列表已保存: 50ETF_option_contracts.csv")

    # 获取所有有效合约代码集合
    valid_codes = set(contracts['ts_code'].unique())

    # 2. 加载已有数据（断点续传）
    existing_dates = load_existing()

    # 3. 按日期区间批量拉取（不指定ts_code，拉全量再过滤）
    chunks = list(date_chunks(START_DATE, END_DATE, CHUNK_DAYS))
    pending = [(s, e) for s, e in chunks if s not in existing_dates or e not in existing_dates]

    # 简化：逐段拉取，已存在的日期段跳过
    pending = [(s, e) for s, e in chunks
               if not existing_dates or not all(
                   d in existing_dates for d in [s[:4] + s[4:6] + s[6:] for s in [s]][:1])]

    # 更简单的方式：按已覆盖日期重新计算待拉取区间
    if existing_dates:
        all_dates = set()
        for s, e in chunks:
            d0 = datetime.strptime(s, '%Y%m%d')
            d1 = datetime.strptime(e, '%Y%m%d')
            while d0 <= d1:
                all_dates.add(d0.strftime('%Y%m%d'))
                d0 += timedelta(days=1)
        missing = sorted(all_dates - existing_dates)
        if not missing:
            log("数据已完整，无需拉取")
            return
        # 从missing重新构造连续区间
        pending = []
        seg_start = missing[0]
        seg_end = missing[0]
        for d in missing[1:]:
            if datetime.strptime(d, '%Y%m%d') - datetime.strptime(seg_end, '%Y%m%d') <= timedelta(days=CHUNK_DAYS):
                seg_end = d
            else:
                pending.append((seg_start, seg_end))
                seg_start = d
                seg_end = d
        pending.append((seg_start, seg_end))
        # 合并为更大的chunk (最多CHUNK_DAYS天)
        merged = []
        for s, e in pending:
            d0 = datetime.strptime(s, '%Y%m%d')
            d1 = datetime.strptime(e, '%Y%m%d')
            while d0 <= d1:
                nxt = min(d0 + timedelta(days=CHUNK_DAYS - 1), d1)
                merged.append((d0.strftime('%Y%m%d'), nxt.strftime('%Y%m%d')))
                d0 = nxt + timedelta(days=1)
        pending = merged
    else:
        pending = chunks

    log(f"计划拉取 {len(pending)} 个日期段")

    # 4. 拉取数据
    new_rows = 0
    for i, (s, e) in enumerate(pending):
        try:
            df = pro.opt_daily(
                start_date=s, end_date=e,
                fields='ts_code,trade_date,open,high,low,close,settle,vol,amount,oi'
            )
            if df is None or df.empty:
                log(f"[{i+1}/{len(pending)}] {s}-{e}: 无数据")
                time.sleep(SLEEP)
                continue

            # 只保留50ETF期权合约
            df = df[df['ts_code'].isin(valid_codes)].copy()

            if df.empty:
                log(f"[{i+1}/{len(pending)}] {s}-{e}: 过滤后无50ETF数据")
                time.sleep(SLEEP)
                continue

            # 追加写入
            write_header = not os.path.exists(OUTPUT_CSV)
            df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False, encoding='utf-8-sig')
            new_rows += len(df)

            pct = (i + 1) / len(pending) * 100
            log(f"[{i+1}/{len(pending)}] {s}-{e}: {len(df)}行 ({pct:.0f}%)")

            time.sleep(SLEEP)

        except Exception as ex:
            log(f"[{i+1}/{len(pending)}] {s}-{e} 失败: {ex}")
            time.sleep(3)

    # 5. 去重、排序
    if os.path.exists(OUTPUT_CSV) and new_rows > 0:
        log("去重排序中...")
        final = pd.read_csv(OUTPUT_CSV, dtype={'trade_date': str, 'ts_code': str})
        before = len(final)
        final = final.drop_duplicates(subset=['ts_code', 'trade_date'])
        final = final.sort_values(['ts_code', 'trade_date'])
        final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        log(f"保存完成: {len(final)}行 (去重删除{before - len(final)}行)")
        log(f"合约数: {final['ts_code'].nunique()}, 日期范围: {final['trade_date'].min()} ~ {final['trade_date'].max()}")
        log(f"文件: {os.path.abspath(OUTPUT_CSV)}")
    elif new_rows == 0:
        log("无新数据，文件未变更")
    else:
        log("未获取到任何数据，请检查Token与网络")


if __name__ == '__main__':
    main()
