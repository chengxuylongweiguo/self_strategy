import pandas as pd
from demo.CommodityFuturesC import CommodityFuturesC, Params
from pythongo.backtesting.engine import run
from pythongo.backtesting.models import Config
from datetime import datetime
main_contract_mapping = {#期货品种映射字典
        datetime(2023, 12, 14, 0, 0, 0): "IM2401",
        datetime(2024, 1, 19, 0, 0, 0): "IM2402",
        datetime(2024, 2, 7, 0, 0, 0): "IM2403",
        datetime(2024, 3, 15, 0, 0, 0): "IM2404",
        datetime(2024, 4, 19, 0, 0, 0): "IM2406",
        datetime(2024, 6, 21, 0, 0, 0): "IM2407",
        datetime(2024, 7, 19, 0, 0, 0): "IM2408",
        datetime(2024, 8, 15, 0, 0, 0): "IM2409",
        datetime(2024, 9, 20, 0, 0, 0): "IM2410",
        datetime(2024, 10, 18, 0, 0, 0): "IM2412",
        datetime(2024, 12, 20, 0, 0, 0): "IM2503",
        datetime(2025, 3, 21, 0, 0, 0): "IM2506",
        datetime(2025, 6, 20, 0, 0, 0): "IM2509"
    }

# 转换为Series并排序（用于查找）
mapping_series = pd.Series(main_contract_mapping)
mapping_series.index = pd.to_datetime(mapping_series.index)  # 确保是 pd.Timestamp
mapping_series = mapping_series.sort_index()

# 查找期货标的函数
def get_main_contract(ts: pd.Timestamp) -> str:
    ts = ts.normalize()  # 去掉时间部分
    mask = mapping_series.index <= ts
    if not mask.any():
        raise ValueError(f"No contract found for date {ts}")
    return mapping_series[mask].iloc[-1]

#生成时间段
def generate_signal_segments_with_buffer(
    trade_day_excel_path: str,
    timestamps_parquet_path: str,
    buffer_days: int = 3):
    # 读取交易日列表
    df_trade = pd.read_excel(trade_day_excel_path)
    trade_days = pd.to_datetime(df_trade["日期"].drop_duplicates()).sort_values()
    trade_days = [d.normalize() for d in trade_days]

    # 读取信号时间戳列表
    df_ts = pd.read_parquet(timestamps_parquet_path)
    timestamps = pd.to_datetime(df_ts["timestamps"]).sort_values()

    segments = []

    for ts in timestamps:
        ts_day = ts.normalize()
        # 找信号对应的交易日索引
        idx = next((i for i, d in enumerate(trade_days) if d >= ts_day), None)
        if idx is None:
            continue

        # 起始日向前推 buffer 天
        start_idx = max(0, idx - buffer_days)
        start_day = trade_days[start_idx]

        # 结束日为信号日的下一个交易日
        if idx + 1 < len(trade_days):
            end_day = trade_days[idx + 1]
        else:
            end_day = trade_days[idx]  # 信号是最后一个交易日

        segments.append((ts,start_day, end_day))

    return segments

 
if __name__ == "__main__":
    backtesting_config = Config(
            access_key="TcsoQ27yhG9QrLmJK2N4Lk",
            access_secret="AxP9GqyBKU+JhiqzhgotpLajNTvlGX5ct4GEYvjTfqY=")
    segments = generate_signal_segments_with_buffer("000852.SH.xlsx", "timestamps.parquet")
    balance = 5000000
    segments_list = segments
    print(segments_list)
    for ts,start_day,end_day in segments_list:
        
        futures_code = get_main_contract(ts)
        params = Params(
            exchange="CFFEX",
            instrument_id=futures_code,
            timestamps=ts
        )

        a = run(
            config=backtesting_config,
            strategy_cls=CommodityFuturesC(),
            strategy_params=params,
            start_date=start_day.strftime('%Y-%m-%d'),
            end_date=end_day.strftime('%Y-%m-%d'),
            initial_capital=balance
        )
        balance = a.dynamic_rights
