from datetime import datetime
import pandas as pd
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
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")

class State(BaseState):
    """状态设置"""
    close_std: float = Field(default=0, title="标准差")
    k: int = Field(default=0, title="状态")
class OptionsDemo1(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()

        self.market_center = MarketCenter()
        self.params_map = Params()
        self.state_map = State()
        self.order_ids: set[int] = set()
        self.index_price: float = 0.0
        self.option_price: float = 0.0
        self.option_code: str = self.params_map.option_code
        self.max_5_percentile = 0
        self.latest_ma_std = 0
        self.start = 6360
        self.end = 6410
        self.key = 0
        self.t = 0
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
    
    @property
    def sub_indicator_data(self) -> dict[str, float]:
        """副图指标"""
        return {
            "std": self.state_map.close_std,
            "M5-0.9":self.max_5_percentile,
            "std_ma":self.latest_ma_std,
        }

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
        expire_date = datetime.strptime(expire_str, "%Y%m%d").date()
        
        # 当前日期和剩余到期时间（年化）
        today = datetime.now().date()
        days_to_expire = (expire_date - today).days
        T = max(days_to_expire, 0) / 365  # 防止为负
        
        # 获取当前期权价格
        option_price =  self.option_price
        r = 0.015  # 默认无风险利率
        
        #输出行权价
        parts = option_code.split('-')
        strike_price = ''.join(filter(str.isdigit, parts[-1]))
        strike_price = float(strike_price)
        #输出期权类型
        option_type = "CALL" if "-C-" in option_code else ("PUT" if "-P-" in option_code else None)
        # 创建 Option 实例并计算希腊值
        option_temp = Option(
            option_type=option_type,#期权代码
            underlying_price=self.index_price,#标的指数价格
            strike_price=strike_price,#行权价
            time_to_expire=T,#剩余到期时间
            risk_free=r,#无风险利率
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
            count=-2)
        self.output(kline)
        self.index_price = kline[-1]['close']
        self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.params_map.instrument_id)#订阅行情
        self.sub_market_data(exchange='SSE',instrument_id='000852')#订阅行情

        self.kline_generator = KLineGenerator(
            real_time_callback=self.real_time_callback,
            callback=self.callback,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.kline_style
        )
        
        #初始化
        self.kline_generator_option = KLineGenerator(
            real_time_callback=self.real_time_callback_option,
            callback=self.callback_option,
            exchange=self.params_map.exchange,
            instrument_id=self.option_code,
            style=self.params_map.kline_style
        )

        self.kline_generator_index = KLineGenerator(
            real_time_callback=self.real_time_callback_index,
            callback=self.callback_index,
            exchange="SSE",
            instrument_id="000852",
            style='M5'
        )
        self.kline_generator.push_history_data()
        super().on_start()
    
    def on_stop(self) -> None:
        self.output('策略暂停')
        super().on_stop()

    def on_tick(self, tick: TickData) -> None:
        """收到行情 tick 推送"""
        #self.output(tick.instrument_id)
        self.t = 1
        super().on_tick(tick)
        self.kline_generator_option.tick_to_kline(tick)
        self.kline_generator.tick_to_kline(tick)
        self.kline_generator_index.tick_to_kline(tick)

    def callback(self, kline: KLineData) -> None:
        close_std_array = self.kline_generator.producer.std(timeperiod=6,array=True)
        valid_array = close_std_array[~np.isnan(close_std_array)]

        if valid_array.size > 100:
            self.max_5_percentile = np.percentile(valid_array, 95)

        valid_series = pd.Series(valid_array)
        ma_std = valid_series.rolling(window=120).mean()  # 120期移动平均

        # 3. 取最后一个作为副图指标（或整段用于画线）
        self.latest_ma_std = ma_std.iloc[-1] if not ma_std.empty else 0.0
        self.state_map.close_std = close_std_array[-1] if close_std_array.size > 0 else 0

        if self.state_map.k == 0 and self.t == 1:
            strike_rounded = int(math.floor(self.index_price / 100.0)-1) * 100 
            ym_str = datetime.now().strftime('%y%m')
            self.option_code = f"MO{ym_str}-P-{strike_rounded}"
            self.state_map.k = 1
            self.output('期权代码：',self.option_code)
            #更新KLineGenerator类
            self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.option_code)#订阅行情
            self.kline_generator_option = KLineGenerator(
            real_time_callback=self.real_time_callback_option,
            callback=self.callback_option,
            exchange=self.params_map.exchange,
            instrument_id=self.option_code,
            style=self.params_map.kline_style)
            
            
            #获取self.option_price当时价格，因为要出现价格抖动才会更新
            option_kline = self.market_center.get_kline_data(
                exchange=self.params_map.exchange,
                instrument_id=self.option_code,
                style="M1",
                count=-2)
            self.output('期权代码：',self.option_code,' 期权价格：',self.option_price)
            self.option_price = option_kline[-1]['close']
            self.output()
            price = self.option_price + self.params_map.pay_up
            
            self.order_ids.add(
                self.send_order(
                    exchange=self.params_map.exchange,
                    instrument_id=self.option_code,
                    volume=40,
                    price=price,
                    order_direction="buy"
                )
            )

        # 1. 获取当前网格目标仓位
        futures_price = kline.close
        key, target_price = min(self.rules.items(), key=lambda x: abs(x[1] - futures_price))
        signal_price = 0
        if key != self.key and self.state_map.k == 1:
            self.key = key
            self.output("期权价格:",self.option_price,"指数价格:",self.index_price)
            delta,gamma = self.calculate_option_greeks(self.option_code)
            
            option_pos = self.get_position(self.option_code).net_position # 获取当前option净仓位

            futures_delta = -delta*option_pos + 2*option_pos*gamma
            futures_position = math.ceil(futures_delta / 2) 

            current_pos = self.get_position(self.params_map.instrument_id).net_position # 2. 获取当前futures净仓位
            delta_position = futures_position - current_pos
            
            if delta_position > 0 and futures_price < target_price:
                # 需要加仓
                price = signal_price = kline.close + self.params_map.pay_up
                self.order_ids.add(
                    self.send_order(
                        exchange=self.params_map.exchange,
                        instrument_id=self.params_map.instrument_id,
                        volume=delta_position,
                        price=price,
                        order_direction="buy"
                    )
                )
            elif delta_position < 0 and futures_price > target_price:
                # 需要减仓
                price = kline.close - self.params_map.pay_up
                signal_price = -price
                self.order_ids.add(
                    self.auto_close_position(
                        exchange=self.params_map.exchange,
                        instrument_id=self.params_map.instrument_id,
                        volume=abs(delta_position),
                        price=price,
                        order_direction="sell"
                    )
                )

        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
            **self.sub_indicator_data
        })
 
    def real_time_callback_option(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""
        self.callback_option(kline)
    
    def real_time_callback_index(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""   
        self.callback_index(kline)

    def real_time_callback(self, kline: KLineData) -> None:
        self.callback(kline)

    def callback_index(self, kline: KLineData) -> None:
        self.index_price = kline.close
        #self.output(' 指数价格：',self.index_price)
        
    def callback_option(self, kline: KLineData) -> None:
        self.option_price = kline.close
        #self.output(' 期权价格：',self.option_price)
        
    
    