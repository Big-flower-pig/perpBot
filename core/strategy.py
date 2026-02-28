"""
PerpBot 策略引擎模块

实现交易策略逻辑：
- 技术指标计算
- 趋势分析
- 支撑阻力位识别
- 信号生成
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd
import numpy as np

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.helpers import safe_float


class Signal(Enum):
    """交易信号"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class Confidence(Enum):
    """信心程度"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Trend(Enum):
    """趋势方向"""

    STRONG_UP = "强势上涨"
    STRONG_DOWN = "强势下跌"
    WEAK_UP = "弱势上涨"
    WEAK_DOWN = "弱势下跌"
    SIDEWAYS = "震荡整理"


@dataclass
class TechnicalIndicators:
    """技术指标数据"""

    # 移动平均线
    sma_5: float = 0
    sma_20: float = 0
    sma_50: float = 0
    ema_12: float = 0
    ema_26: float = 0

    # MACD
    macd: float = 0
    macd_signal: float = 0
    macd_histogram: float = 0

    # RSI
    rsi: float = 50

    # 布林带
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    bb_position: float = 0.5  # 当前价格在布林带中的位置 (0-1)

    # 成交量
    volume: float = 0
    volume_ma: float = 0
    volume_ratio: float = 1.0

    # 支撑阻力
    resistance: float = 0
    support: float = 0
    dynamic_resistance: float = 0
    dynamic_support: float = 0


@dataclass
class TrendAnalysis:
    """趋势分析结果"""

    short_term: str = "未知"  # 短期趋势
    medium_term: str = "未知"  # 中期趋势
    overall: Trend = Trend.SIDEWAYS
    macd_trend: str = "neutral"  # bullish / bearish / neutral
    rsi_level: str = "中性"  # 超买 / 超卖 / 中性
    strength: float = 0  # 趋势强度 (-100 到 100)


@dataclass
class SignalResult:
    """信号结果"""

    signal: Signal
    reason: str
    confidence: Confidence
    stop_loss: float
    take_profit: float
    indicators: Optional[TechnicalIndicators] = None
    trend: Optional[TrendAnalysis] = None
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyEngine:
    """策略引擎

    实现技术分析策略

    Example:
        engine = StrategyEngine()
        signal = engine.analyze(kline_data, current_price)
        if signal.signal == Signal.BUY:
            # 执行买入
            ...
    """

    def __init__(self):
        self._logger = get_logger("strategy")
        self._signal_history: List[SignalResult] = []

    def calculate_indicators(self, kline_data) -> TechnicalIndicators:
        """计算技术指标

        Args:
            kline_data: K线数据，可以是 DataFrame 或 List[Dict]

        Returns:
            TechnicalIndicators 对象
        """
        try:
            indicators = TechnicalIndicators()

            # 转换为 DataFrame
            if isinstance(kline_data, list):
                df = pd.DataFrame(kline_data)
            else:
                df = kline_data

            # 移动平均线
            df["sma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
            df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
            df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

            indicators.sma_5 = safe_float(df["sma_5"].iloc[-1])
            indicators.sma_20 = safe_float(df["sma_20"].iloc[-1])
            indicators.sma_50 = safe_float(df["sma_50"].iloc[-1])

            # 指数移动平均线
            df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

            indicators.ema_12 = safe_float(df["ema_12"].iloc[-1])
            indicators.ema_26 = safe_float(df["ema_26"].iloc[-1])

            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

            indicators.macd = safe_float(df["macd"].iloc[-1])
            indicators.macd_signal = safe_float(df["macd_signal"].iloc[-1])
            indicators.macd_histogram = safe_float(df["macd_histogram"].iloc[-1])

            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            indicators.rsi = safe_float(df["rsi"].iloc[-1], 50)

            # 布林带
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

            indicators.bb_upper = safe_float(df["bb_upper"].iloc[-1])
            indicators.bb_middle = safe_float(df["bb_middle"].iloc[-1])
            indicators.bb_lower = safe_float(df["bb_lower"].iloc[-1])

            # 计算价格在布林带中的位置
            current_price = safe_float(df["close"].iloc[-1])
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                indicators.bb_position = (
                    current_price - indicators.bb_lower
                ) / bb_range
            else:
                indicators.bb_position = 0.5

            # 成交量
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]

            indicators.volume = safe_float(df["volume"].iloc[-1])
            indicators.volume_ma = safe_float(df["volume_ma"].iloc[-1])
            indicators.volume_ratio = safe_float(df["volume_ratio"].iloc[-1], 1.0)

            # 支撑阻力位
            df["resistance"] = df["high"].rolling(window=20).max()
            df["support"] = df["low"].rolling(window=20).min()

            indicators.resistance = safe_float(df["resistance"].iloc[-1])
            indicators.support = safe_float(df["support"].iloc[-1])
            indicators.dynamic_resistance = indicators.bb_upper
            indicators.dynamic_support = indicators.bb_lower

            return indicators

        except Exception as e:
            self._logger.error(f"计算技术指标失败: {e}")
            return TechnicalIndicators()

    def analyze_trend(
        self,
        kline_data,
        indicators: TechnicalIndicators,
    ) -> TrendAnalysis:
        """分析趋势

        Args:
            kline_data: K线数据，可以是 DataFrame 或 List[Dict]
            indicators: 技术指标

        Returns:
            TrendAnalysis 对象
        """
        try:
            analysis = TrendAnalysis()

            # 转换为 DataFrame
            if isinstance(kline_data, list):
                df = pd.DataFrame(kline_data)
            else:
                df = kline_data

            current_price = safe_float(df["close"].iloc[-1])

            # 短期趋势（基于MA20）
            if current_price > indicators.sma_20:
                analysis.short_term = "上涨"
            else:
                analysis.short_term = "下跌"

            # 中期趋势（基于MA50）
            if current_price > indicators.sma_50:
                analysis.medium_term = "上涨"
            else:
                analysis.medium_term = "下跌"

            # MACD 趋势
            if indicators.macd > indicators.macd_signal:
                analysis.macd_trend = "bullish"
            else:
                analysis.macd_trend = "bearish"

            # RSI 状态
            if indicators.rsi > 70:
                analysis.rsi_level = "超买"
            elif indicators.rsi < 30:
                analysis.rsi_level = "超卖"
            else:
                analysis.rsi_level = "中性"

            # 综合趋势判断
            if analysis.short_term == "上涨" and analysis.medium_term == "上涨":
                analysis.overall = Trend.STRONG_UP
            elif analysis.short_term == "下跌" and analysis.medium_term == "下跌":
                analysis.overall = Trend.STRONG_DOWN
            elif analysis.short_term == "上涨":
                analysis.overall = Trend.WEAK_UP
            elif analysis.short_term == "下跌":
                analysis.overall = Trend.WEAK_DOWN
            else:
                analysis.overall = Trend.SIDEWAYS

            # 趋势强度计算
            strength = 0
            # MACD 贡献
            if indicators.macd_histogram > 0:
                strength += 30
            else:
                strength -= 30

            # RSI 贡献
            strength += (indicators.rsi - 50) * 0.6

            # 布林带位置贡献
            strength += (indicators.bb_position - 0.5) * 40

            analysis.strength = max(-100, min(100, strength))

            return analysis

        except Exception as e:
            self._logger.error(f"趋势分析失败: {e}")
            return TrendAnalysis()

    def generate_signal(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators,
        trend: TrendAnalysis,
        current_position: Optional[Dict] = None,
    ) -> SignalResult:
        """生成交易信号

        Args:
            df: K线数据
            indicators: 技术指标
            trend: 趋势分析
            current_position: 当前持仓

        Returns:
            SignalResult 对象
        """
        try:
            current_price = safe_float(df["close"].iloc[-1])
            previous_price = safe_float(df["close"].iloc[-2])
            price_change = ((current_price - previous_price) / previous_price) * 100

            signal = Signal.HOLD
            reason = ""
            confidence = Confidence.MEDIUM

            # 止损止盈计算
            stop_loss_pct = get_config("risk.stop_loss_percent", 3.0)
            take_profit_pct = get_config("risk.take_profit_percent", 5.0)

            # 趋势判断逻辑
            if trend.overall == Trend.STRONG_UP:
                # 强势上涨趋势
                if indicators.rsi < 70:  # 未超买
                    signal = Signal.BUY
                    reason = f"强势上涨趋势，RSI={indicators.rsi:.1f}，MACD看涨"
                    confidence = Confidence.HIGH
                elif indicators.rsi > 80:
                    signal = Signal.HOLD
                    reason = "强势上涨但严重超买，建议观望"
                    confidence = Confidence.LOW
                else:
                    signal = Signal.HOLD
                    reason = "强势上涨但接近超买，谨慎操作"
                    confidence = Confidence.MEDIUM

            elif trend.overall == Trend.STRONG_DOWN:
                # 强势下跌趋势
                if indicators.rsi > 30:  # 未超卖
                    signal = Signal.SELL
                    reason = f"强势下跌趋势，RSI={indicators.rsi:.1f}，MACD看跌"
                    confidence = Confidence.HIGH
                elif indicators.rsi < 20:
                    signal = Signal.HOLD
                    reason = "强势下跌但严重超卖，可能反弹"
                    confidence = Confidence.LOW
                else:
                    signal = Signal.SELL
                    reason = "强势下跌趋势延续"
                    confidence = Confidence.MEDIUM

            elif trend.overall == Trend.WEAK_UP:
                # 弱势上涨
                if indicators.macd > indicators.macd_signal and indicators.rsi < 65:
                    signal = Signal.BUY
                    reason = "弱势上涨，MACD金叉确认"
                    confidence = Confidence.MEDIUM
                else:
                    signal = Signal.HOLD
                    reason = "上涨动能不足，建议观望"
                    confidence = Confidence.LOW

            elif trend.overall == Trend.WEAK_DOWN:
                # 弱势下跌
                if indicators.macd < indicators.macd_signal and indicators.rsi > 35:
                    signal = Signal.SELL
                    reason = "弱势下跌，MACD死叉确认"
                    confidence = Confidence.MEDIUM
                else:
                    signal = Signal.HOLD
                    reason = "下跌动能不足，建议观望"
                    confidence = Confidence.LOW

            else:
                # 震荡整理
                # 检查是否突破布林带
                if indicators.bb_position > 0.8:
                    signal = Signal.BUY
                    reason = "突破布林带上轨，可能启动上涨"
                    confidence = Confidence.MEDIUM
                elif indicators.bb_position < 0.2:
                    signal = Signal.SELL
                    reason = "跌破布林带下轨，可能启动下跌"
                    confidence = Confidence.MEDIUM
                else:
                    signal = Signal.HOLD
                    reason = "震荡整理，无明确方向"
                    confidence = Confidence.LOW

            # 成交量确认
            if signal != Signal.HOLD:
                if indicators.volume_ratio < 1.0:
                    # 成交量不足，降低信心
                    if confidence == Confidence.HIGH:
                        confidence = Confidence.MEDIUM
                    elif confidence == Confidence.MEDIUM:
                        confidence = Confidence.LOW
                    reason += "（成交量偏低）"
                elif indicators.volume_ratio > 2.0:
                    # 成交量放大，增强信心
                    confidence = Confidence.HIGH
                    reason += "（成交量放大）"

            # 计算止损止盈价格
            if signal == Signal.BUY:
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
            elif signal == Signal.SELL:
                stop_loss = current_price * (1 + stop_loss_pct / 100)
                take_profit = current_price * (1 - take_profit_pct / 100)
            else:
                stop_loss = 0
                take_profit = 0

            result = SignalResult(
                signal=signal,
                reason=reason,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
                trend=trend,
            )

            # 记录信号
            self._signal_history.append(result)
            if len(self._signal_history) > 100:
                self._signal_history.pop(0)

            # 记录日志
            self._logger.signal(
                signal=signal.value,
                symbol=get_config("trading.symbol", "UNKNOWN"),
                confidence=confidence.value,
                reason=reason,
            )

            return result

        except Exception as e:
            self._logger.error(f"生成信号失败: {e}")
            return SignalResult(
                signal=Signal.HOLD,
                reason=f"信号生成异常: {e}",
                confidence=Confidence.LOW,
                stop_loss=0,
                take_profit=0,
            )

    def analyze(
        self,
        kline_data: List[Dict],
        current_position: Optional[Dict] = None,
    ) -> SignalResult:
        """完整分析流程

        Args:
            kline_data: K线数据列表
            current_position: 当前持仓

        Returns:
            SignalResult 对象
        """
        try:
            # 转换为 DataFrame
            df = pd.DataFrame(kline_data)

            # 计算技术指标
            indicators = self.calculate_indicators(df)

            # 分析趋势
            trend = self.analyze_trend(df, indicators)

            # 生成信号
            signal = self.generate_signal(df, indicators, trend, current_position)

            return signal

        except Exception as e:
            self._logger.error(f"分析失败: {e}")
            return SignalResult(
                signal=Signal.HOLD,
                reason=f"分析异常: {e}",
                confidence=Confidence.LOW,
                stop_loss=0,
                take_profit=0,
            )

    @property
    def signal_history(self) -> List[SignalResult]:
        """获取信号历史"""
        return self._signal_history.copy()

    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计"""
        if not self._signal_history:
            return {}

        signals = [s.signal.value for s in self._signal_history]
        return {
            "total": len(signals),
            "buy_count": signals.count("BUY"),
            "sell_count": signals.count("SELL"),
            "hold_count": signals.count("HOLD"),
            "last_signal": self._signal_history[-1].signal.value
            if self._signal_history
            else None,
        }
