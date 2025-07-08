from pythongo.base import BaseParams, Field
from pythongo.classdef import KLineData, TickData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator
 
 
class Params(BaseParams):
    """参数映射模型"""
    exchange: str = Field(default="CFFEX", title="交易所代码")
    instrument_id: str = Field(default="IM2507", title="合约代码")
    fast_period: int = Field(default=5, title="快均线周期", ge=2)
    slow_period: int = Field(default=20, title="慢均线周期")
    kline_style: KLineStyleType = Field(default="M1", title="K 线周期")
 
 
class DemoKLine(BaseStrategy):
    """我的第二个策略"""
    def __init__(self) -> None:
        super().__init__()
        self.params_map = Params()
 
        self.fast_ma = 0.0
        self.slow_ma = 0.0
 
    @property
    def main_indicator_data(self) -> dict[str, float]:
        """主图指标"""
        start = 6270
        end = 6330
        steps = 3
        price_step = (end - start) / steps
        rules = {}
        for i in range(steps + 1):
            price = round(start + i * price_step, 3)
            rules[f'P{i}'] = price
        return rules
    
    def on_start(self) -> None:
        self.kline_generator = KLineGenerator(
            real_time_callback=self.real_time_callback, #用于在 当前这根K线形成过程中，每次 tick 推送触发的处理逻辑（比如实时计算指标）
            callback=self.callback, #每根 完整K线 收盘时触发的回调函数，通常用于下单、记录信号
            exchange=self.params_map.exchange,#交易所代码
            instrument_id=self.params_map.instrument_id,#合约代码
            style=self.params_map.kline_style #k线周期
        )
        self.kline_generator.push_history_data()
        super().on_start()
 
    def on_stop(self) -> None:
        super().on_stop()
 
    def on_tick(self, tick: TickData) -> None:
        """收到行情 tick 推送"""
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)
 
    def calc_indicator(self) -> None:
        """计算指标数据"""
        self.slow_ma = self.kline_generator.producer.sma(
            timeperiod=self.params_map.slow_period
        )
 
        self.fast_ma = self.kline_generator.producer.sma(
            timeperiod=self.params_map.fast_period
        )
 
    def callback(self, kline: KLineData) -> None:
        """接受 K 线回调"""
        self.calc_indicator()
 
        self.widget.recv_kline({
            "kline": kline,
            **self.main_indicator_data
        })
 
    def real_time_callback(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""
        self.callback(kline)