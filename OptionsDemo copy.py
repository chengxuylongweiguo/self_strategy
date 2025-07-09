from datetime import datetime
import math
import re
from typing import Literal
import numpy as np
from scipy.stats import norm
from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType,MarketCenter
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator
from pythongo.option import Option


class Params(BaseParams):
    """参数设置"""
    exchange: str = Field(default="CFFEX", title="交易所")
    instrument_id: str = Field(default="IM2507", title="合约代码")
    option_code: str = Field(default="MO2507-P-6400", title="期权代码")
    steps: int = Field(default=5, title="网格层数", ge=1)
    pay_up: float = Field(default=0.2, title="滑价超价")
    kline_style: KLineStyleType = Field(default="M1", title="K线周期")


class State(BaseState):
    """状态设置"""
    bought: bool = Field(default=False, title="是否已经买入")


class OptionsDemo(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.market_center = MarketCenter()
        self.params_map = Params()
        self.state_map = State()
        self.order_ids: set[int] = set()
        self.index_price: float = 0.0
        self.option_price: float = 0.0
        self.option_code: str = self.params_map.option_code
        self.start = 6370
        self.start = 6360
        self.end = 6390
        self.key = 0
        self.k = 0
        self.rules = self.main_indicator_data
        self.position_map = {'P0': 5,
                                'P1': 4,
                                'P2': 3,
                                'P3': 2,
                                'P4': 1
                            }
        
    @property
    def main_indicator_data(self) -> dict[str, float]:
        """主图指标"""
        steps = self.params_map.steps
        price_step = (self.end - self.start) / (steps-1)
        rules = {}
        for i in range(steps):
            price = round(self.start + i * price_step, 3)
            rules[f'P{i}'] = price
        return rules
    
    #计算gamma、delta
    def calculate_option_greeks(self, option_code: str) -> tuple[float, float]:
        """
        计算指定期权的 Delta 和 Gamma。
        参数：
            option_code (str): 期权合约代码
            strike_price (float): 行权价（K）
        返回：
            tuple: (delta, gamma)
        """
        # 获取期权合约的到期日字符串，并转换为日期对象
        expire_str = self.get_instrument_data(
            exchange=self.params_map.exchange,
            instrument_id=option_code
        ).expire_date
        expire_date = datetime.strptime(expire_str, "%Y-%m-%d").date()
        
        # 当前日期和剩余到期时间（年化）
        today = datetime.now().date()
        days_to_expire = (expire_date - today).days
        T = max(days_to_expire, 0) / 365  # 防止为负
        
        # 获取当前期权价格
        option_kline_list = self.market_center.get_kline_data(
            exchange=self.params_map.exchange,
            instrument_id=option_code,
            style=self.params_map.kline_style,
            count=-1
        )
        if not option_kline_list:
            raise ValueError(f"未能获取 {option_code} 的K线数据")

        option_price = option_kline_list[-1].close
        r = 0.015  # 默认无风险利率
        
        #输出行权价
        parts = option_code.split('-')
        strike_price = ''.join(filter(str.isdigit, parts[-1]))
        strike_price = float(strike_price)
        # 创建 Option 实例并计算希腊值
        option_temp = Option(
            option_type=option_code,#期权代码
            underlying_price=self.index_price,#标的指数价格
            K=strike_price,#行权价
            t=T,#剩余到期时间
            r=r,#无风险利率
            market_price=option_price,#期权价格
            dividend_rate=0 #股息率
        )
        delta = option_temp.bs_delta()
        gamma = option_temp.bs_gamma()
        return delta, gamma

    def on_start(self) -> None:
        kline = self.market_center.get_kline_data(
            exchange="SSE",
            instrument_id="000852",
            style="M1",
            count=-1
        )
        self.output(kline)
        self.index_price = kline[-1]['close']
        self.kline_generator = KLineGenerator(
            real_time_callback=self.real_time_callback,
            callback=self.callback,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.kline_style
        )

        self.kline_generator_index = KLineGenerator(
            real_time_callback=self.real_time_callback_index,
            callback=self.callback_index,
            exchange="SSE",
            instrument_id="000852",
            style='M1'
        )
        
        self.kline_generator_m5 = KLineGenerator(
            callback=self.callback_m5,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style='M5'
        )
        self.kline_generator.push_history_data()
        self.kline_generator_m5.push_history_data()
        super().on_start()
    
    def on_stop(self) -> None:
        super().on_stop()

    def on_tick(self, tick: TickData) -> None:
        """收到行情 tick 推送"""
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)
        self.kline_generator_index.tick_to_kline(tick)
        self.kline_generator_m5.tick_to_kline(tick)

    def callback(self, kline: KLineData) -> None:
        if self.k == 0:
            strike_rounded = int(math.floor(self.index_price / 100.0)-1) * 100 
            ym_str = datetime.now().strftime('%y%m')
            self.option_code = f"MO{ym_str}-P-{strike_rounded}"
            self.k = 1
            self.kline_generator_option = KLineGenerator(
            real_time_callback=self.real_time_callback_option,
            callback=self.callback_option,
            exchange=self.params_map.exchange,
            instrument_id=self.option_code,
            style='M1')
            price = self.option_price + self.params_map.pay_up
            self.output('期权代码：',self.option_code,' 期权价格：',price)
            self.order_ids.add(
                self.send_order(
                    exchange=self.params_map.exchange,
                    instrument_id=self.option_code,
                    volume=20,
                    price=price,
                    order_direction="buy"
                )
            )

        #delta,gamma = self.calculate_option_greeks(self.option_code)
        stock_price = kline.close

        # 1. 获取当前网格目标仓位
        key, target_price = min(self.rules.items(), key=lambda x: abs(x[1] - stock_price))
        target_pos = self.position_map.get(key, 0)
        #stock_delta = -delta*self.positions_dict['long'][option_code]['amount'] + 2*self.positions_dict['long'][option_code]['amount']*gamma
        # 2. 获取当前净仓位
        current_pos = self.get_position(self.params_map.instrument_id).net_position
        # 3. 差额 = 目标仓位 - 当前仓位
        delta = target_pos - current_pos
        signal_price = 0
        if delta > 0:
            # 需要加仓
            price = signal_price = kline.close + self.params_map.pay_up
            self.order_ids.add(
                self.send_order(
                    exchange=self.params_map.exchange,
                    instrument_id=self.params_map.instrument_id,
                    volume=delta,
                    price=price,
                    order_direction="buy"
                )
            )
        elif delta < 0:
            # 需要减仓
            price = kline.close - self.params_map.pay_up
            signal_price = -price
            self.order_ids.add(
                self.auto_close_position(
                    exchange=self.params_map.exchange,
                    instrument_id=self.params_map.instrument_id,
                    volume=abs(delta),
                    price=price,
                    order_direction="sell"
                )
            )
        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data
        })
 
    def real_time_callback(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""
        self.callback(kline)
    
    def real_time_callback_index(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""   
        self.callback_index(kline)

    def real_time_callback_option(self, kline: KLineData) -> None:
        self.callback_option(kline)

    def callback_index(self, kline: KLineData) -> None:
        self.index_price = kline.close

    def callback_option(self, kline: KLineData) -> None:
        self.option_price = kline.close

    def callback_m5(self, kline: KLineData) -> None:
        close_std_array = self.kline_generator_m5.producer.std(timeperiod=6,array=True)
        
        
    
    