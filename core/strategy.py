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

    # ===== 新增: 专业指标 =====
    # ATR (平均真实波幅) - 用于动态止损
    atr: float = 0
    atr_percent: float = 0  # ATR 占价格百分比

    # ADX (平均趋向指数) - 用于趋势强度判断
    adx: float = 0
    plus_di: float = 0  # +DI
    minus_di: float = 0  # -DI

    # 市场状态
    market_regime: str = "UNKNOWN"  # TRENDING / RANGING / TRANSITIONAL


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

            # ===== 新增: 计算 ATR =====
            atr = self._calculate_atr(df)
            indicators.atr = atr
            current_price = safe_float(df["close"].iloc[-1])
            if current_price > 0:
                indicators.atr_percent = (atr / current_price) * 100

            # ===== 新增: 计算 ADX =====
            adx, plus_di, minus_di = self._calculate_adx(df)
            indicators.adx = adx
            indicators.plus_di = plus_di
            indicators.minus_di = minus_di

            # ===== 新增: 判断市场状态 =====
            indicators.market_regime = self._detect_market_regime(indicators)

            return indicators

        except Exception as e:
            self._logger.error(f"计算技术指标失败: {e}")
            return TechnicalIndicators()

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算 ATR (Average True Range)

        ATR 用于衡量市场波动性，是设置止损止盈的重要参考。

        Args:
            df: K线数据
            period: 计算周期

        Returns:
            ATR 值
        """
        try:
            # 计算真实波幅 (True Range)
            high = df["high"]
            low = df["low"]
            close = df["close"]

            prev_close = close.shift(1)

            tr1 = high - low  # 当日最高-最低
            tr2 = abs(high - prev_close)  # 当日最高-昨收
            tr3 = abs(low - prev_close)  # 当日最低-昨收

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 计算 ATR (指数移动平均)
            atr = tr.ewm(span=period, adjust=False).mean().iloc[-1]

            return safe_float(atr, 0)

        except Exception as e:
            self._logger.debug(f"ATR 计算失败: {e}")
            return 0

    def _calculate_adx(
        self, df: pd.DataFrame, period: int = 14
    ) -> Tuple[float, float, float]:
        """计算 ADX (Average Directional Index)

        ADX 用于判断趋势强度:
        - ADX > 25: 强趋势
        - ADX < 20: 无趋势或震荡
        - +DI > -DI: 多头趋势
        - -DI > +DI: 空头趋势

        Args:
            df: K线数据
            period: 计算周期

        Returns:
            (ADX, +DI, -DI)
        """
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # 计算 +DM 和 -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            # +DM: 当日高点-昨日高点 > 昨日低点-当日低点 且 > 0
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            # -DM: 昨日低点-当日低点 > 当日高点-昨日高点 且 > 0
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # 计算 True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 平滑处理
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

            # 计算 DX 和 ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean().iloc[-1]

            return (
                safe_float(adx, 0),
                safe_float(plus_di.iloc[-1], 0),
                safe_float(minus_di.iloc[-1], 0),
            )

        except Exception as e:
            self._logger.debug(f"ADX 计算失败: {e}")
            return 0, 0, 0

    def _detect_market_regime(self, indicators: TechnicalIndicators) -> str:
        """检测市场状态

        根据ADX和ATR判断当前市场处于趋势还是震荡状态。

        优化版: ADX 是主要判断标准，ATR 作为辅助确认

        Args:
            indicators: 技术指标

        Returns:
            市场状态: TRENDING / RANGING / TRANSITIONAL
        """
        adx = indicators.adx
        atr_pct = indicators.atr_percent
        plus_di = indicators.plus_di
        minus_di = indicators.minus_di

        # ADX > 30: 明显趋势 (降低阈值，ADX > 30 已足够强)
        # 不再要求 ATR% > 1%，因为加密货币 ATR 通常较低
        if adx > 30:
            return "TRENDING"

        # ADX > 20 且 DI 方向明确: 趋势形成中
        if adx > 20 and abs(plus_di - minus_di) > 10:
            return "TRENDING"

        # ADX < 15: 明显震荡
        if adx < 15:
            return "RANGING"

        # 其他情况: 过渡状态
        return "TRANSITIONAL"

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
        """生成交易信号 (优化版: 基于市场状态的智能信号)

        核心改进:
        1. 根据 ADX 判断市场状态 (趋势/震荡)
        2. 趋势市使用趋势策略，震荡市使用均值回归策略
        3. 使用 ATR 动态计算止损止盈

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

            # ===== 核心改进: 基于市场状态的策略选择 =====
            market_regime = indicators.market_regime
            adx = indicators.adx

            # 记录市场状态
            self._logger.debug(
                f"市场状态: {market_regime}, ADX={adx:.1f}, "
                f"ATR%={indicators.atr_percent:.2f}%"
            )

            # ===== 策略1: 趋势市场 (ADX > 25) =====
            if market_regime == "TRENDING":
                signal, reason, confidence = self._trend_strategy(
                    indicators, trend, current_price
                )

            # ===== 策略2: 震荡市场 (ADX < 20) =====
            elif market_regime == "RANGING":
                signal, reason, confidence = self._range_strategy(
                    indicators, current_price
                )

            # ===== 策略3: 过渡状态 (20 <= ADX <= 25) =====
            else:
                signal, reason, confidence = self._transitional_strategy(
                    indicators, trend, current_price
                )

            # ===== 成交量确认 =====
            if signal != Signal.HOLD:
                if indicators.volume_ratio < 1.0:
                    # 成交量不足，降低信心
                    if confidence == Confidence.HIGH:
                        confidence = Confidence.MEDIUM
                    elif confidence == Confidence.MEDIUM:
                        confidence = Confidence.LOW
                    reason += " (量能不足)"
                elif indicators.volume_ratio > 2.0:
                    # 成交量放大，增强信心
                    confidence = Confidence.HIGH
                    reason += " (量能强劲)"

            # ===== 核心改进: 使用 ATR 动态止损止盈 =====
            stop_loss, take_profit = self._calculate_dynamic_stops(
                current_price=current_price,
                atr=indicators.atr,
                signal=signal,
                risk_reward_ratio=2.0,  # 盈亏比 1:2
            )

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

    def _trend_strategy(
        self,
        indicators: TechnicalIndicators,
        trend: TrendAnalysis,
        current_price: float,
    ) -> Tuple[Signal, str, Confidence]:
        """趋势市场策略

        在明确的趋势市场中，顺势交易。

        Args:
            indicators: 技术指标
            trend: 趋势分析
            current_price: 当前价格

        Returns:
            (信号, 原因, 信心程度)
        """
        signal = Signal.HOLD
        reason = ""
        confidence = Confidence.MEDIUM

        # 判断趋势方向
        is_uptrend = indicators.plus_di > indicators.minus_di
        is_downtrend = indicators.minus_di > indicators.plus_di

        if is_uptrend:
            # 多头趋势 - 寻找做多机会
            if indicators.rsi < 70:
                # RSI 未超买，可以入场
                if trend.overall in [Trend.STRONG_UP, Trend.WEAK_UP]:
                    signal = Signal.BUY
                    if indicators.adx > 40:
                        confidence = Confidence.HIGH
                        reason = f"强多头趋势 (ADX={indicators.adx:.1f}, +DI>{'-DI'})"
                    else:
                        confidence = Confidence.MEDIUM
                        reason = f"多头趋势确立 (ADX={indicators.adx:.1f})"
                else:
                    signal = Signal.BUY
                    confidence = Confidence.LOW
                    reason = "趋势形成中，轻仓试探"
            else:
                signal = Signal.HOLD
                reason = f"多头趋势但 RSI 超买 ({indicators.rsi:.1f})，等待回调"

        elif is_downtrend:
            # 空头趋势 - 寻找做空机会
            if indicators.rsi > 30:
                # RSI 未超卖，可以入场
                if trend.overall in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
                    signal = Signal.SELL
                    if indicators.adx > 40:
                        confidence = Confidence.HIGH
                        reason = f"强空头趋势 (ADX={indicators.adx:.1f}, -DI>{'+DI'})"
                    else:
                        confidence = Confidence.MEDIUM
                        reason = f"空头趋势确立 (ADX={indicators.adx:.1f})"
                else:
                    signal = Signal.SELL
                    confidence = Confidence.LOW
                    reason = "趋势形成中，轻仓试探"
            else:
                signal = Signal.HOLD
                reason = f"空头趋势但 RSI 超卖 ({indicators.rsi:.1f})，等待反弹"

        else:
            signal = Signal.HOLD
            reason = "趋势方向不明确，观望"

        return signal, reason, confidence

    def _range_strategy(
        self,
        indicators: TechnicalIndicators,
        current_price: float,
    ) -> Tuple[Signal, str, Confidence]:
        """震荡市场策略

        在震荡市场中，使用均值回归策略，低买高卖。

        Args:
            indicators: 技术指标
            current_price: 当前价格

        Returns:
            (信号, 原因, 信心程度)
        """
        signal = Signal.HOLD
        reason = ""
        confidence = Confidence.MEDIUM

        # 使用布林带和 RSI 判断超买超卖
        bb_pos = indicators.bb_position
        rsi = indicators.rsi

        # 超卖区域 - 买入信号
        if bb_pos < 0.2 and rsi < 35:
            signal = Signal.BUY
            if rsi < 25:
                confidence = Confidence.HIGH
                reason = f"深度超卖 (RSI={rsi:.1f}, BB位置={bb_pos:.1%})"
            else:
                confidence = Confidence.MEDIUM
                reason = f"超卖区域 (RSI={rsi:.1f})"

        # 超买区域 - 卖出信号
        elif bb_pos > 0.8 and rsi > 65:
            signal = Signal.SELL
            if rsi > 75:
                confidence = Confidence.HIGH
                reason = f"深度超买 (RSI={rsi:.1f}, BB位置={bb_pos:.1%})"
            else:
                confidence = Confidence.MEDIUM
                reason = f"超买区域 (RSI={rsi:.1f})"

        # 中性区域 - 观望
        else:
            signal = Signal.HOLD
            reason = f"震荡中继 (RSI={rsi:.1f}, BB位置={bb_pos:.1%})"

        return signal, reason, confidence

    def _transitional_strategy(
        self,
        indicators: TechnicalIndicators,
        trend: TrendAnalysis,
        current_price: float,
    ) -> Tuple[Signal, str, Confidence]:
        """过渡状态策略 (优化版: 更积极的信号生成)

        在趋势形成过程中，结合多种指标寻找入场机会。

        改进点:
        1. 放宽 RSI 阈值，从 25/75 调整为 35/65
        2. 结合 MACD 和布林带确认信号
        3. 考虑短期趋势方向

        Args:
            indicators: 技术指标
            trend: 趋势分析
            current_price: 当前价格

        Returns:
            (信号, 原因, 信心程度)
        """
        signal = Signal.HOLD
        reason = ""
        confidence = Confidence.LOW

        rsi = indicators.rsi
        bb_pos = indicators.bb_position
        macd_hist = indicators.macd_histogram

        # 多头信号条件 (放宽条件)
        bullish_signals = 0
        bullish_reasons = []

        if rsi < 40:  # 放宽: RSI < 40 视为偏低
            bullish_signals += 1
            if rsi < 30:
                bullish_reasons.append(f"RSI超卖({rsi:.0f})")
            else:
                bullish_reasons.append(f"RSI偏低({rsi:.0f})")

        if bb_pos < 0.35:  # 放宽: 布林带下轨区域
            bullish_signals += 1
            bullish_reasons.append(f"BB下轨区({bb_pos:.0%})")

        if macd_hist > 0:  # MACD 柱状图正值
            bullish_signals += 1
            bullish_reasons.append("MACD转正")
        elif indicators.macd > indicators.macd_signal:
            bullish_signals += 0.5
            bullish_reasons.append("MACD金叉形成")

        # 空头信号条件 (放宽条件)
        bearish_signals = 0
        bearish_reasons = []

        if rsi > 60:  # 放宽: RSI > 60 视为偏高
            bearish_signals += 1
            if rsi > 70:
                bearish_reasons.append(f"RSI超买({rsi:.0f})")
            else:
                bearish_reasons.append(f"RSI偏高({rsi:.0f})")

        if bb_pos > 0.65:  # 放宽: 布林带上轨区域
            bearish_signals += 1
            bearish_reasons.append(f"BB上轨区({bb_pos:.0%})")

        if macd_hist < 0:  # MACD 柱状图负值
            bearish_signals += 1
            bearish_reasons.append("MACD转负")
        elif indicators.macd < indicators.macd_signal:
            bearish_signals += 0.5
            bearish_reasons.append("MACD死叉形成")

        # 结合趋势方向确认
        trend_boost = 0
        if trend.overall in [Trend.STRONG_UP, Trend.WEAK_UP]:
            trend_boost = 1
        elif trend.overall in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
            trend_boost = -1

        # 生成信号
        final_bullish = bullish_signals + (trend_boost * 0.5 if trend_boost > 0 else 0)
        final_bearish = bearish_signals + (
            abs(trend_boost) * 0.5 if trend_boost < 0 else 0
        )

        if final_bullish >= 2:
            signal = Signal.BUY
            reason = ", ".join(bullish_reasons[:2])
            if final_bullish >= 3:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW
        elif final_bearish >= 2:
            signal = Signal.SELL
            reason = ", ".join(bearish_reasons[:2])
            if final_bearish >= 3:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW
        else:
            signal = Signal.HOLD
            reason = f"信号不明确 (多:{bullish_signals:.1f}/空:{bearish_signals:.1f})"

        return signal, reason, confidence

    def _calculate_dynamic_stops(
        self,
        current_price: float,
        atr: float,
        signal: Signal,
        risk_reward_ratio: float = 2.0,
        atr_multiplier: float = 2.0,
    ) -> Tuple[float, float]:
        """基于 ATR 的动态止损止盈计算

        核心原理:
        - 止损距离 = ATR × 倍数 (通常 1.5-2.0)
        - 止盈距离 = 止损距离 × 盈亏比

        Args:
            current_price: 当前价格
            atr: ATR 值
            signal: 交易信号
            risk_reward_ratio: 盈亏比
            atr_multiplier: ATR 倍数

        Returns:
            (止损价, 止盈价)
        """
        if signal == Signal.HOLD or atr <= 0:
            return 0, 0

        # 止损距离
        stop_distance = atr * atr_multiplier
        # 止盈距离
        take_profit_distance = stop_distance * risk_reward_ratio

        if signal == Signal.BUY:
            stop_loss = current_price - stop_distance
            take_profit = current_price + take_profit_distance
        else:  # SELL
            stop_loss = current_price + stop_distance
            take_profit = current_price - take_profit_distance

        # 确保止损止盈为正数
        stop_loss = max(stop_loss, 0)
        take_profit = max(take_profit, 0)

        return stop_loss, take_profit

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
