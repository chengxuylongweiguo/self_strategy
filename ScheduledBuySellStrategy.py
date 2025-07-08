from datetime import time
from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import TickData, TradeData, OrderData, KLineData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy


class Params(BaseParams):
    """参数设置"""
    exchange: str = Field(default="CFFEX", title="交易所")
    instrument_id: str = Field(default="MO2507-C-6300", title="合约代码")
    order_volume: int = Field(default=1, title="下单手数", ge=1)
    pay_up: float = Field(default=0.2, title="滑价超价")
    kline_style: KLineStyleType = Field(default="M1", title="K线周期")


class State(BaseState):
    """状态设置"""
    bought: int = Field(default=1, title="是否已经买入")


class ScheduledBuySellStrategy(BaseStrategy):
    """
    每日 15:50 开多一手 IF2507
    每日 15:55 平掉持仓
    """

    def __init__(self) -> None:
        super().__init__()
        self.params_map = Params()
        self.state_map = State()
        self.order_ids: set[int] = set()

    def on_tick(self, tick: TickData) -> None:
        super().on_tick(tick)

        t: time = tick.datetime.time()

        # 判断是否是 15:50（买入）
        if t.hour == 15 and t.minute == 31 and not self.state_map.bought:
            position = self.get_position(self.params_map.instrument_id)
            self.state_map.bought = position.net_position
            

        # 判断是否是 15:55（卖出）
        if t.hour == 15 and t.minute == 33 and self.state_map.bought:
            position = self.get_position(self.params_map.instrument_id)
            self.state_map.bought = position.net_position
                

    def on_trade(self, trade: TradeData, log: bool = False) -> None:
        super().on_trade(trade, log)
        self.order_ids.discard(trade.order_id)

    def on_order_cancel(self, order: OrderData) -> None:
        super().on_order_cancel(order)
        self.order_ids.discard(order.order_id)

    def on_start(self) -> None:
        super().on_start()

    def on_stop(self) -> None:
        super().on_stop()
    
