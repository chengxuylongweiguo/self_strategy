from datetime import datetime
import math
from typing import Literal
import numpy as np
from scipy.stats import norm
from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType,MarketCenter
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator


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
        self.start = 6360
        self.end = 6390
        self.key = 0
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
    
    #计算delta
    def calculate_delta(self,S, K, T, r, sigma, option_type='call'):
        """
        计算欧式股指期权价格（Black模型，适用于现金交割）
        :param S: 标的指数当前价格
        :param K: 期权执行价
        :param T: 剩余期限（年）
        :param r: 无风险利率（年化）
        :param sigma: 隐含波动率（年化）
        :param option_type: 'call'（看涨）或 'put'（看跌）
        :return: 期权价格
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            delta = np.exp(-r * T) * norm.cdf(d1)
        else:
            delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
        return delta
    #计算gamma
    def calculate_gamma_independent(self,S, K, T, r, sigma, is_future=True, q=0.0):
        """
        计算欧式股指期权价格（Black模型，适用于现金交割）
        :param S: 标的指数当前价格
        :param K: 期权执行价
        :param T: 剩余期限（年）
        :param r: 无风险利率（年化）
        :param sigma: 隐含波动率（年化）
        :param option_type: 'call'（看涨）或 'put'（看跌）
        :return: 期权价格
        """
        # 边界条件处理
        if S <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        # 添加极小值避免数值问题
        T = max(T, 1e-6)
        sigma = max(sigma, 1e-4)
        # 处理期货期权特例
        if is_future:
            q = r
        # 计算d1
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        # 计算标准正态分布的概率密度函数
        phi_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
        # 计算Gamma
        gamma = np.exp(-q * T) * phi_d1 / (S * sigma * np.sqrt(T))
        return gamma


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

    def callback(self, kline: KLineData) -> None:
        kline_list = self.market_center.get_kline_data(
                                        exchange="CFFEX",
                                        instrument_id="IM2507",
                                        style=self.params_map.kline_style,
                                        count=-1
                                    )
        option_code = "MO2507-C-6400"

        stock_price = kline.close
        # 1. 获取当前网格目标仓位
        key, target_price = min(self.rules.items(), key=lambda x: abs(x[1] - stock_price))
        target_pos = self.position_map.get(key, 0)
        
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
    

    
    