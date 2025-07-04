from pythongo.base import BaseParams, Field
from pythongo.classdef import KLineData, TickData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator
 
 
class Params(BaseParams):
    """参数映射模型"""
    exchange: str = Field(default="", title="交易所代码")
    instrument_id: str = Field(default="", title="合约代码")
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
        return {
            f"MA{self.params_map.fast_period}": self.fast_ma,
            f"MA{self.params_map.slow_period}": self.slow_ma,
        }
 
    def on_start(self) -> None:
        self.kline_generator = KLineGenerator(
            real_time_callback=self.real_time_callback,
            callback=self.callback,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.kline_style
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