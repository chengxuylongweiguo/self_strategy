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
#

class Params(BaseParams):
    """参数设置"""
    exchange: str = Field(default="SHFE", title="交易所")
    instrument_id: str = Field(default="ao2508", title="合约代码")
    signal_manual: str = Field(default="False", title="手动触发")
    iv_strat: float = Field(default=0.0, title="波动起始价格")
    quantile: int = Field(default=95, title="信号分位数阈值")
    middle_value: float = Field(default=0.0, title="网格中间值", ge=0)
    max_value: float = Field(default=0.0, title="网格值", ge=0)
    steps: int = Field(default=12, title="网格层数", ge=1)
    pay_up: float = Field(default=0.2, title="滑价超价")
    order_type: bool = Field(default=False, title="市价单")
    ym_str: str = Field(default="2508", title="到期月份")
    options:str = Field(default="", title="期权代码")
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")
    

class State(BaseState):
    """状态设置"""
    close_std: float = Field(default=0, title="标准差")
    k: int = Field(default=0, title="状态")

class CommodityFuturesB(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.market_center = MarketCenter()
        self.params_map = Params()
        self.state_map = State()
        self.order_ids: set[int] = set() #
        self.option_price: float = 0.0 #期权价格
        self.option_volume: int = 0
        self.futures_price: float = 0.0#用于初始化网格
        self.option_code: str = self.params_map.options#期权代码
        self.index_code: str = "option"#当月期货代码
        self.index_price: float #当月期货价格
        self.open_signal: Union[bool, str] = False #开仓信号
        self.order_futures: int = 0 #期货是否下单 废弃
        self.rules: dict = {} #网格字典
        self.iv_start_close = 0 #波动开始的价格
        self.max_5_percentile = 0 #标准差分位数值
        self.latest_ma_std = 0 #标准差均值  
        self.min_value = 0 #网格上限
        self.max_value = 0 #网格下限
        self.key = None #状态更新
        self.order_dict: dict = {}#无法下市价单就在tick中下单
        self.interval_map = {#行权价间隔
    "au": 8,
    "ao": 50,
    "br": 200,
    "rb": 50,
    "ru": 250,
    "m": 50,
    "p": 100,
    "lh": 200,
    "eg": 50,
    "eb": 100,
    "UR": 20,
    "PX": 100,
    "CF": 200,
    "SF": 50,
    "SM": 100,
    "AP": 100,
    "sc": 10,
    "lc": 1000,
    "si": 100,
    "pp": 100,
    "l":100,
    'si':100,
}
    
    @property
    def main_indicator_data(self) -> dict[str, float]:
        """主图指标"""
        rules = {}
        steps = self.params_map.steps
        if self.min_value == 0:
            for i in range(steps):
                rules[f'P{i}'] = self.futures_price
            return rules
        price_step = (self.max_value - self.min_value) / (steps)
        for i in range(steps+1):
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

    #计算网格信号
    def detect_rule_cross(self, rules: dict, price_last: float, price_now: float):
        """
        判断是否穿越 rules 中的价格点，返回距离当前价格最近的穿越点的持仓量（volume）。
        没有穿越则返回 None。
        """
        crossed = []
        sorted_prices = sorted(rules.keys())
        for price in sorted_prices:
            if price_last < price <= price_now:
                self.output(f"上穿{price}")
                crossed.append(price)
            elif price_last > price >= price_now:
                self.output(f"下穿{price}")
                crossed.append(price)
        if crossed:
            # 找到距离当前价格最近的穿越点，返回对应的 volume
            nearest_price = min(crossed, key=lambda x: abs(x - price_now))
            return rules[nearest_price]
        else:
            return None
                
    #生成期权代码
    def get_option_code(self,index_code,option_type,strike_rounded):
        """ 依次输入标的首字母，期权类型，行权价 """
        code = index_code
        ym_str = self.params_map.ym_str
        if self.params_map.exchange == "SHFE":#上期所
            option_code = f"{code}{ym_str}{option_type}{strike_rounded}"
            return option_code
        elif self.params_map.exchange == "CZCE":#郑商所
            option_code = f"{code}{ym_str}{option_type}{strike_rounded}"
            return option_code
        elif self.params_map.exchange == "DCE":#大商所
            option_code = f"{code}{ym_str}-{option_type}-{strike_rounded}"
            return option_code
        elif self.params_map.exchange == "GFEX":#广期所
            option_code = f"{code}{ym_str}-{option_type}-{strike_rounded}"
            return option_code
        elif self.params_map.exchange == "INE":#能源中心
            option_code = f"{code}{ym_str}{option_type}{strike_rounded}"
            return option_code
        else:
            raise ValueError("交易所输入错误！") 

    #进场信号 
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
        if not expire_str:
            raise ValueError("expire_str 为空，无法解析期权到期日")
        expire_date = datetime.strptime(expire_str, "%Y%m%d").date()
        
        # 当前日期和剩余到期时间（年化）
        today = datetime.now().date()
        days_to_expire = (expire_date - today).days
        T = max(days_to_expire, 0) / 365  # 防止为负
        option_price =  self.option_price #期权价格
        r = 0.015  # 默认无风险利率
        strike_price = int(''.join(re.findall(r'\d+', option_code))[4:])#输出行权价
    
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
        code = re.match(r"([a-zA-Z]+)", self.params_map.instrument_id).group(1)#获取前面的期货品种代码
        self.index_code = f"{code}{self.params_map.ym_str}"
        if self.params_map.options != "":#如果输入了期权代码
            if self.params_map.exchange == "CZCE":
                match = int(''.join(re.findall(r'\d+', self.params_map.options))[:3])
                self.index_code = f"{code}{match}"
            else:
                match = int(''.join(re.findall(r'\d+', self.params_map.options))[:4])
                self.index_code = f"{code}{match}"
        
        self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.index_code)
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
        if tick.last_price != 0:
            if self.order_dict and tick.instrument_id == self.order_dict['instrument_id']: #当推送标的的tick时下单
                if self.order_dict['order_direction'] == 'buy':
                    price = tick.ask_price3 if tick.ask_price3 != 0 else tick.ask_price1 # 买入 -> 使用卖一价
                elif self.order_dict['order_direction'] == 'sell':
                    price = tick.bid_price3 if tick.bid_price3 != 0 else tick.bid_price1 # 卖出 -> 使用买一价
                if self.order_dict['direction'] == 'buy':
                    self.order_ids.add(
                        self.send_order(
                                exchange=self.params_map.exchange,
                                instrument_id=self.order_dict['instrument_id'],
                                volume=self.order_dict['volume'],
                                price=price,
                                market=self.params_map.order_type,
                                order_direction=self.order_dict['order_direction']
                            )
                    )
                    self.order_dict = {}
                
                elif self.order_dict['direction'] == 'sell':
                    self.order_ids.add(
                        self.auto_close_position(
                            exchange=self.params_map.exchange,
                            instrument_id=self.order_dict['instrument_id'],
                            volume=self.order_dict['volume'],
                            price=price,
                            market=self.params_map.order_type,
                            order_direction=self.order_dict['order_direction']
                        )
                    )
                    self.order_dict = {}
                
            if tick.instrument_id == self.option_code : #当推送期权的tick时
                self.option_price = tick.last_price #更新期权价格
                
            if tick.instrument_id == self.index_code:
                self.index_price = tick.last_price  #更新对应期权到期日当月期货指数价格

            self.kline_generator.tick_to_kline(tick)

    #报单回调
    def on_order(self, order: OrderData) -> None:
        self.output(f'合约代码：{order.instrument_id} 订单状态：{order.status} 成交数量：{order.traded_volume}')
        super().on_order(order)
        if order.traded_volume == order.total_volume:#完全成交时删除委托记录
            if order.order_id in self.order_ids:
                self.order_ids.remove(order.order_id)
                self.output(f"未成交委托列表：{self.order_ids}")

        elif order.status == '已撤销' :##撤单时删除委托记录
            if order.order_id in self.order_ids:
                self.order_ids.remove(order.order_id)
                self.output(f"{order.order_id}撤单{self.order_ids}")

    def callback(self, kline: KLineData) -> None:
        self.futures_price = kline.close #没出现信号和历史推送时画网格用的,计算gamma
        signal_price = 0
        if self.open_signal == False:
            iv_signal = self.analyze_volatility_structure() 
            if self.trading: # 可交易状态
                if (self.params_map.signal_manual == 'rise' or self.params_map.signal_manual == 'fall') and self.params_map.iv_strat != 0:
                    iv_signal = self.params_map.signal_manual #手动设置信号
                    self.iv_start_close = self.params_map.iv_strat #手动设置波动开始时的行情
                self.output(iv_signal)
                if iv_signal == 'fall' or iv_signal == 'rise':
                    new_price = self.futures_price
                    middle_close = (self.iv_start_close + new_price) / 2 #中点价格
                    self.min_value = 2*self.futures_price - middle_close #不一定是最小值，可能是最大值 下面同理但不影响网格生成
                    self.max_value = middle_close  
                    rules = self.main_indicator_data #fall就从小到大，rise就从大到小
                    
            
                #暴跌信号处理
                if iv_signal == 'fall':   
                    #如果没有手动输入就，自动选择期权
                    if self.params_map.options == "":
                        code = re.match(r'^[A-Za-z]+', self.params_map.instrument_id).group()
                        strike_interval = self.interval_map[code]
                        otm_strike_rounded = int((math.floor(self.index_price / strike_interval)-1) * strike_interval) 
                        self.option_code = self.get_option_code(code,'P',otm_strike_rounded) #月份需要手动输入
                    
                    #订阅行情
                    self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.option_code) #订阅虚值行情
                    self.kline_generator_option = KLineGenerator(
                    callback=self.callback_option,
                    exchange=self.params_map.exchange,
                    instrument_id=self.option_code,
                    style='M1')
                    self.kline_generator_option.push_history_data()
                    
                    self.open_signal = iv_signal #更新状态 防止重复触发入场
                    signal_price = self.futures_price
                    
                    future_pos = 50
                    delta,gamma = self.calculate_option_greeks(self.option_code,self.index_price,'PUT')
                    option_pos = future_pos / ((-delta + 2*gamma))
                    option_pos = math.ceil(option_pos)   
                    self.option_volume = option_pos
                    #生成价格持仓量字典
                    volume_step = (-option_pos) / (self.params_map.steps)
                    self.rules = {rules[f'P{i}']:round(option_pos + i * volume_step) for i in range(self.params_map.steps + 1)}
                    
                    self.output('期权代码：',self.option_code,'买入数量：',option_pos,'买入价格：',self.option_price)
                    #买入期权
                    self.order_dict = {"instrument_id":self.option_code,"volume":option_pos,'order_direction':"buy",'direction':"buy"}
                        
                #暴涨信号处理
                elif iv_signal == 'rise':
                    #如果没有手动输入就，自动选择期权
                    if self.params_map.options == "":
                        code = re.match(r'^[A-Za-z]+', self.params_map.instrument_id).group()
                        strike_interval = self.interval_map[code]
                        otm_strike_rounded = int((math.ceil(self.index_price / strike_interval) + 1) * strike_interval) 
                        self.option_code = self.get_option_code(code,'C',otm_strike_rounded) #月份需要手动输入
                    
                    #订阅行情
                    self.sub_market_data(exchange=self.params_map.exchange,instrument_id=self.option_code) #订阅虚值行情
                    self.kline_generator_option = KLineGenerator(
                    callback=self.callback_option,
                    exchange=self.params_map.exchange,
                    instrument_id=self.option_code,
                    style='M1')
                    self.kline_generator_option.push_history_data()
                      
                    self.open_signal = iv_signal #更新状态 防止重复触发入场
                    signal_price = self.futures_price
                    
                    future_pos = 50
                    delta,gamma = self.calculate_option_greeks(self.option_code,self.index_price,'CALL')
                    option_pos = future_pos / ((delta + 20*gamma))
                    option_pos = math.ceil(option_pos)
                    self.option_volume = option_pos
                    #生成价格持仓量字典
                    volume_step = (-option_pos) / (self.params_map.steps)
                    self.rules = {rules[f'P{i}']:round(option_pos + i * volume_step) for i in range(self.params_map.steps + 1)}
                    
                    self.output('期权代码：',self.option_code,'买入数量：',option_pos,'买入价格：',self.option_price)
                    #买入期权
                    self.order_dict = {"instrument_id":self.option_code,"volume":option_pos,'order_direction':"buy",'direction':"buy"}
                
            else:#历史推送时执行
                if iv_signal == 'fall':
                    signal_price = -self.futures_price
                    iv_signal = False
                elif iv_signal == 'rise':
                    signal_price = self.futures_price
                    iv_signal = False

        #手动设置网格
        elif self.params_map.max_value != 0 and self.params_map.middle_value != 0:
            self.min_value = self.params_map.max_value #不一定是最小值，可能是最大值 下面同理但不影响网格生成
            self.max_value = self.params_map.middle_value 
            self.rules = self.main_indicator_data
        
        #更新副图指标值
        close_array = self.kline_generator.producer.close
        close_series = pd.Series(close_array)
        std_series = close_series.rolling(6).std()
        ma_std_series = std_series.rolling(120).mean()
        valid_std = std_series.dropna()
        self.max_5_percentile = np.percentile(valid_std.values, self.params_map.quantile ) if len(valid_std) > 100 else 0
        self.state_map.close_std = std_series.iloc[-1] if not std_series.empty else 0
        self.latest_ma_std = ma_std_series.iloc[-1] if not ma_std_series.empty else 0

        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
            **self.sub_indicator_data
        })
        
    def real_time_callback(self, kline: KLineData) -> None:
        signal_price = 0 #初始化买卖图像信号
        if self.open_signal == 'fall' and self.get_position(self.option_code).net_position == self.option_volume:
            if self.order_futures == 0:
                #计算期货持仓量
                delta,gamma = self.calculate_option_greeks(self.option_code,self.index_price,'PUT') 
                option_pos = self.get_position(self.option_code).net_position # 获取当前option净仓位
                future_pos = option_pos * (-delta + 20*gamma)
                future_pos = math.ceil(future_pos) 
                #下单
                self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":future_pos,'order_direction':"buy",'direction':"buy"}
                self.order_futures = future_pos

            rules_volume = self.detect_rule_cross(self.rules,self.futures_price,kline.close)
            self.futures_price = kline.close #顺序不能变
            
            if rules_volume is not None and self.key != rules_volume and self.get_position(self.params_map.instrument_id).net_position == self.order_futures:
                self.key = rules_volume

                current_pos = self.get_position(self.params_map.instrument_id).net_position
                delta_position = rules_volume - current_pos

                for order_id in self.order_ids:#全部撤单再进行调仓
                        self.cancel_order(order_id)
                        
                if delta_position > 0 :#加仓
                    signal_price = self.futures_price
                    self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":delta_position,'order_direction':"buy",'direction':"buy"}
                elif delta_position < 0 :#减仓
                    signal_price = -self.futures_price
                    self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":abs(delta_position),'order_direction':"sell",'direction':"sell"}

        if self.open_signal == 'rise' and self.get_position(self.option_code).net_position == self.option_volume:
            if self.order_futures == 0:
                #计算期货持仓量
                delta,gamma = self.calculate_option_greeks(self.option_code,self.index_price,'PUT') 
                option_pos = self.get_position(self.option_code).net_position # 获取当前option净仓位
                future_pos = option_pos * (delta + 20*gamma)
                future_pos = math.ceil(future_pos) 
                #下单
                self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":future_pos,'order_direction':"buy",'direction':"buy"}
                self.order_futures = future_pos

            rules_volume = self.detect_rule_cross(self.rules,self.futures_price,kline.close)
            self.futures_price = kline.close #顺序不能变
            
            if rules_volume is not None and self.key != rules_volume and self.get_position(self.params_map.instrument_id).net_position == self.order_futures:
                self.key = rules_volume

                current_pos = -self.get_position(self.params_map.instrument_id).net_position
                delta_position = rules_volume - current_pos

                for order_id in self.order_ids:#全部撤单再进行调仓
                        self.cancel_order(order_id)
                        
                if delta_position > 0 :#加仓
                    signal_price = -self.futures_price
                    self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":delta_position,'order_direction':"sell",'direction':"buy"}
                elif delta_position < 0 :#减仓
                    signal_price = self.futures_price
                    self.order_dict = {"instrument_id":self.params_map.instrument_id,"volume":abs(delta_position),'order_direction':"buy",'direction':"sell"}
                             
        """接受 K 线回调"""
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
            **self.sub_indicator_data})
    
    def callback_option(self, kline: KLineData) -> None:
        self.option_price = kline.close
        #self.output(' 期权价格：',self.option_price)

    

    

        
    
    