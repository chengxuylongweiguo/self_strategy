from typing import Literal
from pythongo.base import BaseParams, BaseState, BaseStrategy, Field
from pythongo.classdef import OrderData
from pythongo.core import MarketCenter
from WindPy import w
w.start()
 
class Params(BaseParams):
    """参数映射模型"""
    exchange: str = Field(default="CFFEX", title="交易所代码")
    instrument_id: str = Field(default="IM2507", title="合约代码")
    order_price: int | float = Field(default=0, title="报单价格")
    order_volume: int = Field(default=1, title="报单手数")
    order_direction: Literal["buy", "sell"] = Field(default="buy", title="报单方向")
 
 
class State(BaseState):
    """状态映射模型"""
    order_id: int | None = Field(default=None, title="报单编号")
 
 
class DemoTest(BaseStrategy):
    """我的第一个策略"""
    def __init__(self) -> None:
        super().__init__()
        self.market_center = MarketCenter()
        self.params_map = Params()
        self.state_map = State()
 
    def on_order(self, order: OrderData) -> None:
        super().on_order(order)
        self.output("报单信息：", order)
 
    def on_start(self) -> None:
        super().on_start()
        kline_list = self.market_center.get_kline_data(
                                        exchange="CFFEX",
                                        instrument_id="IM2507",
                                        style='M1',
                                        count=1
                                    )
        print(kline_list)
        self.state_map.order_id = self.send_order(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=self.params_map.order_volume,
            price=self.params_map.order_price,
            order_direction=self.params_map.order_direction
        )
 
        self.update_status_bar()
 
    def on_stop(self) -> None:
        super().on_stop()
        self.output("我的第一个策略暂停了")