from datetime import datetime
import time
import pandas as pd
import math
import re
from typing import Literal,Union
import numpy as np
from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType,MarketCenter
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator
from pythongo.option import Option
#kline.close输出为零未解决
#期权选择行权价未解决

class Params(BaseParams):
    """参数设置"""
    exchange: str = Field(default="SHFE", title="交易所")
    instrument_id: str = Field(default="ao2508", title="合约代码")
    signal_manual: str = Field(default="False", title="手动触发")
    iv_strat: float = Field(default=0, title="波动起始价格")
    quantile: int = Field(default=95, title="信号分位数阈值")
    middle_value: float = Field(default=0.0, title="网格中间值", ge=1)
    max_value: float = Field(default=0.0, title="网格值", ge=1)
    steps: int = Field(default=12, title="网格层数", ge=1)
    pay_up: float = Field(default=0.2, title="滑价超价")
    ym_str: str = Field(default="2508", title="期权到期月份")
    strike_interval: float = Field(default=1, title="期权行权价间隔")
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")
    

class State(BaseState):
    """状态设置"""
    close_std: float = Field(default=0, title="标准差")
    k: int = Field(default=0, title="状态")

class ProducOptions(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.market_center = MarketCenter()
        self.params_map = Params()
        self.state_map = State()
        self.order_ids: set[int] = set()
        self.option_price: float = 0.0
        self.futures_price: float = 6000.0
        self.atm_option_price: float = 0.0
        self.Transaction: int = 0
        self.option_code: str 
        self.atm_option_code: str 
        self.open_signal: Union[bool, str] = False
        self.rules: dict = {}
        self.iv_start_close = 0 #波动开始的价格
        self.max_5_percentile = 0
        self.latest_ma_std = 0
        self.min_value = 0
        self.max_value = 0
        self.key = 0
          
    @property
    def main_indicator_data(self) -> dict[str, float]:
        """主图指标"""
        rules = {}
        steps = self.params_map.steps
        if self.min_value == 0:
            for i in range(steps):
                rules[f'P{i}'] = self.futures_price
            return rules
        price_step = (self.max_value - self.min_value) / (steps-1)
        for i in range(steps):
            price = round(self.min_value + i * price_step, 3)
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

    #推送信号通知
    """ def push_notice(self,text) -> None:
        # 创建带代理的会话
        import telebot
        import requests
        session = requests.Session()
        session.proxies = {
            'https': 'http://127.0.0.1:10808',
            'http': 'http://127.0.0.1:10808'  # 如果需要同时代理 HTTP 请求
        }
        # 使用会话初始化 Bot
        TOKEN = "7738302353:AAGdFjWI6Wg6ye8eFHnvH7N6zRKarlHUZPY"
        bot = telebot.TeleBot(TOKEN, request_session=session)
        # 测试发送消息
        bot.send_message(chat_id=5436165313, text=text) """

    #进场信号 待修改已写死 记录波动开始价格
    def analyze_volatility_structure(self):
        """
        进场信号，记录波动开始的价格iv_start_close、标准差self.state_map.close_std、标准差均值self.latest_ma_std。
        返回：方向：'rise' or 'fall'
        """
        close_array = self.kline_generator.producer.close
        if len(close_array) < 400 :
            return False # 数据不足时不分析
        close_series = pd.Series(close_array)
        # 1. std、std均线、95分位
        std_series = close_series.rolling(6).std()
        ma_std_series = std_series.rolling(120).mean()
        valid_std = std_series.dropna()

        self.max_5_percentile = np.percentile(valid_std.values, self.params_map.quantile) if len(valid_std) > 100 else 0
        self.state_map.close_std = std_series.iloc[-1] if not std_series.empty else 0
        self.latest_ma_std = ma_std_series.iloc[-1] if not ma_std_series.empty else 0

        signal = self.max_5_percentile < std_series.iloc[-2] and std_series.iloc[-3] < std_series.iloc[-2] and std_series.iloc[-1] < std_series.iloc[-2]  #入场信号
        if not signal :
            return False
        # 2. 检测最近一次std低于均值的位置
        condition = std_series < ma_std_series
        valid_index = condition[condition].index
        start_idx = valid_index[-1]
        self.iv_start_close = close_series.iloc[start_idx]
        new_close = close_series.iloc[-1]

        if abs(new_close - self.iv_start_close) <= self.latest_ma_std * 2:
            return False
        direction = 'rise' if new_close > self.iv_start_close else 'fall'
        return direction

    #计算gamma、delta
    def calculate_option_greeks(self, option_code: str, index_price:float,option_type:str) -> tuple[float, float]:
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
        strike_price = int(''.join(re.findall(r'\d+', option_code))[4:])
    
        # 创建 Option 实例并计算希腊值
        option_temp = Option(
            option_type=option_type,#期权代码
            underlying_price=index_price,#标的指数价格
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
        self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.params_map.instrument_id)#订阅行情
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
        self.unsub_market_data()
        super().on_stop()

    def on_tick(self, tick: TickData) -> None:
        """收到行情 tick 推送"""
        #self.output(tick.instrument_id)
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)
        if self.open_signal != False: #当出现信号的时候才接受tick
            self.kline_generator_option.tick_to_kline(tick)
            self.kline_generator_atm_option.tick_to_kline(tick)

    #报单回调
    def on_order(self, order: OrderData) -> None:
        self.output(f'合约代码：{order.instrument_id} 订单状态：{order.status} 成交数量：{order.traded_volume}')
        super().on_order(order)
        
    def callback(self, kline: KLineData) -> None:
        self.futures_price = kline.close #没出现信号和历史推送时画网格用的
        signal_price = 0
        if self.open_signal == False:
            iv_signal = self.analyze_volatility_structure() 
            if self.trading: # 可交易状态
                if (self.params_map.signal_manual == 'rise' or self.params_map.signal_manual == 'fall') and self.params_map.iv_strat != 0:
                    iv_signal = self.params_map.signal_manual #手动设置信号
                    self.iv_start_close = self.params_map.iv_strat #手动设置波动开始时的行情
                self.output(iv_signal)
                if iv_signal == 'fall' or iv_signal == 'rise':
                    #self.push_notice(f"{self.params_map.instrument_id}出现{iv_signal}信号！")
                    #self.iv_start_close = 6460
                    new_price = kline.close
                    middle_close = (self.iv_start_close + new_price) / 2 #中点价格
                    self.min_value = 2*kline.close - middle_close #不一定是最小值，可能是最大值 下面同理但不影响网格生成
                    self.max_value = middle_close  
                    self.rules = self.main_indicator_data
            else:#历史推送时执行
                if iv_signal == 'fall':
                    signal_price = -kline.close
                    iv_signal = False
                elif iv_signal == 'rise':
                    signal_price = kline.close
                    iv_signal = False
            
            #暴跌信号处理
            if iv_signal == 'fall' and self.trading:
                self.output(self.open_signal)
                #获取对应的看跌期权
                strike_interval = self.params_map.strike_interval
                otm_strike_rounded = int((math.floor(kline.close / strike_interval)-1) * strike_interval) 
                self.output(kline.close,'-',otm_strike_rounded)
                atm_strike_rounded = int((round(kline.close / strike_interval)) * strike_interval)
                
                code = re.match(r'^[A-Za-z]+', self.params_map.instrument_id).group()
                ym_str = self.params_map.ym_str
                self.option_code = f"{code}{ym_str}P{otm_strike_rounded}"
                self.atm_option_code = f"{code}{ym_str}P{atm_strike_rounded}"
                self.output(self.option_code)
                self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.option_code) #订阅虚值行情
                self.kline_generator_option = KLineGenerator(
                real_time_callback=self.real_time_callback_option,
                callback=self.callback_option,
                exchange=self.params_map.exchange,
                instrument_id=self.option_code,
                style='M1')

                self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.atm_option_code) #订阅平值行情
                self.kline_generator_atm_option = KLineGenerator(
                real_time_callback=self.real_time_callback_option,
                callback=self.callback_option,
                exchange=self.params_map.exchange,
                instrument_id=self.atm_option_code,
                style='M1')
                #推送历史 K 线数据到回调
                self.kline_generator_option.push_history_data()
                self.kline_generator_atm_option.push_history_data()
                self.open_signal = iv_signal #更新状态 防止重复触发入场
                time.sleep(3) #等价格更新
                #self.option_price = option_kline[-1]['close']
                #self.atm_option_price = atm_option_kline[-1]['close']
                self.output('虚值期权：',self.option_code,' 平值期权：',self.atm_option_code)
                price = kline.close
                self.order_ids.add(
                    self.send_order(
                        exchange=self.params_map.exchange,
                        instrument_id=self.params_map.instrument_id,
                        volume=10,
                        price=price,
                        order_type='FAK',
                        market=False,
                        order_direction="buy"
                    )
                )
                

            #暴涨信号处理
            if iv_signal == 'rise' and self.trading:
                #获取对应的看涨期权
                self.output(self.open_signal)
                strike_interval = self.params_map.strike_interval
                otm_strike_rounded = int(math.ceil(kline.close / strike_interval) + 1) * strike_interval 
                atm_strike_rounded = int(round(kline.close / strike_interval)) * strike_interval
                code = re.match(r'^[A-Za-z]+', self.params_map.instrument_id).group()
                ym_str = self.params_map.ym_str
                self.option_code = f"{code}{ym_str}C{otm_strike_rounded}" 
                self.atm_option_code = f"{code}{ym_str}C{atm_strike_rounded}"
                
                #订阅行情
                self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.option_code) #订阅虚值行情
                self.kline_generator_option = KLineGenerator(
                real_time_callback=self.real_time_callback_option,
                callback=self.callback_option,
                exchange=self.params_map.exchange,
                instrument_id=self.option_code,
                style='M1')
                
                self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.atm_option_code) #订阅平值行情
                self.kline_generator_option = KLineGenerator(
                real_time_callback=self.real_time_callback_option,
                callback=self.callback_option,
                exchange=self.params_map.exchange,
                instrument_id=self.atm_option_code,
                style='M1')
                #推送历史 K 线数据到回调
                self.kline_generator_option.push_history_data()
                self.kline_generator_atm_option.push_history_data()
                self.open_signal = iv_signal #更新状态 防止重复触发入场
                time.sleep(3) #等价格更新
                #self.option_price = option_kline[-1]['close']
                #self.atm_option_price = atm_option_kline[-1]['close']
                self.output('虚值期权：',self.option_code,' 平值期权：',self.atm_option_code)
                price = kline.close
                self.output(price)
                self.order_ids.add(
                    self.send_order(
                        exchange=self.params_map.exchange,
                        instrument_id=self.params_map.instrument_id,
                        volume=10,
                        price=price,
                        order_type='FAK',
                        market=False,
                        order_direction="sell"
                    )
                )

        elif self.params_map.max_value != 0 and self.params_map.middle_value != 0:
            self.min_value = self.params_map.max_value #不一定是最小值，可能是最大值 下面同理但不影响网格生成
            self.max_value = self.params_map.middle_value 
            self.rules = self.main_indicator_data

        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
            **self.sub_indicator_data
        })
        
    def real_time_callback(self, kline: KLineData) -> None:
        #更新副图指标值
        close_array = self.kline_generator.producer.close
        close_series = pd.Series(close_array)
        std_series = close_series.rolling(6).std()
        ma_std_series = std_series.rolling(120).mean()
        valid_std = std_series.dropna()
        self.max_5_percentile = np.percentile(valid_std.values, self.params_map.quantile ) if len(valid_std) > 100 else 0
        self.state_map.close_std = std_series.iloc[-1] if not std_series.empty else 0
        self.latest_ma_std = ma_std_series.iloc[-1] if not ma_std_series.empty else 0
        
        signal_price = 0 #初始化买卖图像信号
        if self.open_signal == 'fall' and self.get_position(self.params_map.instrument_id).net_position != 0:
            futures_price = kline.close
            key, target_price = min(self.rules.items(), key=lambda x: abs(x[1] - futures_price))
            if key != self.key:
                self.key = key #更新网格状态
                delta,gamma = self.calculate_option_greeks(self.option_code,futures_price,'PUT')
                self.output(futures_price)
                future_pos = self.get_position(self.params_map.instrument_id).net_position # 获取当前option净仓位
                option_pos = future_pos*2 / (-delta + 4*gamma)
                option_pos = math.ceil(option_pos) 

                current_pos = self.get_position(self.option_code).net_position # 2. 获取当前futures净仓位
                delta_position = option_pos - current_pos
                self.output("期权价格:",self.option_price,'delta:',delta,'gamma:',gamma,'仓位变化：',delta_position)
                if delta_position > 0 and futures_price > target_price: #需要加仓 要满足价格小于网格价格
                    price = signal_price = self.option_price
                    self.order_ids.add(
                        self.send_order(
                            exchange=self.params_map.exchange,
                            instrument_id=self.option_code,
                            volume=delta_position,
                            price=price,
                            market=False,
                            order_direction="buy"
                        )
                    )

                elif delta_position < 0 and futures_price < target_price: # 需要减仓 要满足价格大于网格价格
                    price = self.option_price
                    signal_price = -price
                    self.order_ids.add(
                        self.auto_close_position(
                            exchange=self.params_map.exchange,
                            instrument_id=self.option_code,
                            volume=abs(delta_position),
                            price=price,
                            market=False,
                            order_direction="sell"
                        )
                    )

        if self.open_signal == 'rise' and self.get_position(self.params_map.instrument_id).net_position != 0:
            futures_price = kline.close
            key, target_price = min(self.rules.items(), key=lambda x: abs(x[1] - futures_price))
            if key != self.key:
                self.key = key #更新网格状态
                delta,gamma = self.calculate_option_greeks(self.option_code,futures_price,'CALL')

                future_pos = self.get_position(self.params_map.instrument_id).net_position # 获取当前option净仓位
                option_pos = future_pos*2 / (delta + 4*gamma)
                option_pos = math.ceil(option_pos) 

                current_pos = self.get_position(self.option_code).net_position # 2. 获取当前futures净仓位
                delta_position = option_pos - current_pos
                self.output("期权价格:",self.option_price,'delta:',delta,'gamma:',gamma,'仓位变化：',delta_position)
                if delta_position > 0 and futures_price < target_price: #需要加仓 要满足价格小于网格价格
                    price = signal_price = self.option_price + self.params_map.pay_up
                    self.order_ids.add(
                        self.send_order(
                            exchange=self.params_map.exchange,
                            instrument_id=self.option_code,
                            volume=delta_position,
                            price=price,
                            market=False,
                            order_direction="buy"
                        )
                    )
                
                elif delta_position < 0 and futures_price > target_price: # 需要减仓 要满足价格大于网格价格
                    price = self.option_price - self.params_map.pay_up
                    signal_price = -price
                    self.order_ids.add(
                        self.auto_close_position(
                            exchange=self.params_map.exchange,
                            instrument_id=self.option_code,
                            volume=abs(delta_position),
                            price=price,
                            market=False,
                            order_direction="sell"
                        )
                    )

        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
            **self.sub_indicator_data})

    def real_time_callback_option(self, kline: KLineData) -> None:
        """使用收到的实时推送 K 线来计算指标并更新线图"""
        self.callback_option(kline)
    
    def callback_option(self, kline: KLineData) -> None:
        if kline.instrument_id == self.option_code:
            self.option_price = kline.close
        elif kline.instrument_id == self.atm_option_code:
            self.atm_option_price = kline.close
        #self.output(' 期权价格：',self.option_price)

    

        
    
    