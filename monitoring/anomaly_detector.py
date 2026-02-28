"""
PerpBot 异常检测模块

提供市场和交易异常检测功能：
- 价格异常检测
- 成交量异常检测
- 波动率异常检测
- 交易行为异常检测
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import math

from utils.logger import get_logger, TradingLogger
from utils.config import get_config


class AnomalyType(Enum):
    """异常类型"""
    PRICE_OUTLIER = "price_outlier"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    TREND_REVERSAL = "trend_reversal"
    GAP_DETECTION = "gap_detection"
    LIQUIDATION_RISK = "liquidation_risk"


class Severity(Enum):
    """严重程度"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Anomaly:
    """异常事件"""
    anomaly_type: AnomalyType
    severity: Severity
    symbol: str
    description: str
    value: float
    expected_range: tuple
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "description": self.description,
            "value": self.value,
            "expected_range": self.expected_range,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AnomalyDetector:
    """异常检测器

    使用统计方法检测市场异常

    Example:
        detector = AnomalyDetector()

        # 添加数据点
        detector.add_price(price=95000)

        # 检测异常
        anomalies = detector.detect()

        # 检查是否异常
        if detector.is_anomaly(price):
            print("价格异常!")
    """

    def __init__(self):
        self._logger = get_logger("anomaly")
        self._symbol = get_config("trading.symbol", "BTC/USDT:USDT")
        self._lock = threading.Lock()

        # 数据存储
        self._prices: deque = deque(maxlen=500)
        self._volumes: deque = deque(maxlen=500)
        self._returns: deque = deque(maxlen=500)  # 收益率序列

        # 统计缓存
        self._stats_cache: Dict[str, Any] = {}
        self._cache_valid = False

        # 检测配置
        self._config = {
            "z_score_threshold": get_config("monitoring.anomaly.z_score_threshold", 3.0),
            "volatility_window": get_config("monitoring.anomaly.volatility_window", 20),
            "min_data_points": get_config("monitoring.anomaly.min_data_points", 30),
        }

    def add_price(self, price: float, volume: float = 0):
        """添加价格数据点

        Args:
            price: 价格
            volume: 成交量
        """
        with self._lock:
            # 计算收益率
            if len(self._prices) > 0:
                last_price = self._prices[-1]
                if last_price > 0:
                    ret = (price - last_price) / last_price
                    self._returns.append(ret)

            self._prices.append(price)
            if volume > 0:
                self._volumes.append(volume)

            self._cache_valid = False

    def detect(self) -> List[Anomaly]:
        """检测所有类型的异常

        Returns:
            异常列表
        """
        anomalies = []

        with self._lock:
            if len(self._prices) < self._config["min_data_points"]:
                return anomalies

            # 更新统计缓存
            self._update_stats_cache()

            # 检测各类异常
            price_anomaly = self._detect_price_outlier()
            if price_anomaly:
                anomalies.append(price_anomaly)

            volatility_anomaly = self._detect_volatility_spike()
            if volatility_anomaly:
                anomalies.append(volatility_anomaly)

            volume_anomaly = self._detect_volume_anomaly()
            if volume_anomaly:
                anomalies.append(volume_anomaly)

            trend_anomaly = self._detect_trend_reversal()
            if trend_anomaly:
                anomalies.append(trend_anomaly)

            gap_anomaly = self._detect_gap()
            if gap_anomaly:
                anomalies.append(gap_anomaly)

        return anomalies

    def is_anomaly(self, price: float) -> bool:
        """快速检查价格是否异常

        Args:
            price: 当前价格

        Returns:
            是否异常
        """
        with self._lock:
            if len(self._prices) < self._config["min_data_points"]:
                return False

            self._update_stats_cache()

            mean = self._stats_cache.get("price_mean", 0)
            std = self._stats_cache.get("price_std", 0)

            if std <= 0:
                return False

            z_score = abs(price - mean) / std
            return z_score > self._config["z_score_threshold"]

    def _detect_price_outlier(self) -> Optional[Anomaly]:
        """检测价格离群点"""
        if not self._prices:
            return None

        current_price = self._prices[-1]
        mean = self._stats_cache.get("price_mean", 0)
        std = self._stats_cache.get("price_std", 0)

        if std <= 0:
            return None

        z_score = (current_price - mean) / std

        if abs(z_score) > self._config["z_score_threshold"]:
            severity = self._get_severity(abs(z_score))
            direction = "高于" if z_score > 0 else "低于"

            return Anomaly(
                anomaly_type=AnomalyType.PRICE_OUTLIER,
                severity=severity,
                symbol=self._symbol,
                description=f"价格{direction}均值 {abs(z_score):.2f} 个标准差",
                value=current_price,
                expected_range=(mean - 3 * std, mean + 3 * std),
                metadata={"z_score": z_score, "mean": mean, "std": std},
            )

        return None

    def _detect_volatility_spike(self) -> Optional[Anomaly]:
        """检测波动率异常"""
        if len(self._returns) < self._config["volatility_window"]:
            return None

        # 计算近期波动率
        recent_returns = list(self._returns)[-self._config["volatility_window"]:]
        recent_vol = math.sqrt(sum(r ** 2 for r in recent_returns) / len(recent_returns))

        # 计算历史波动率
        all_returns = list(self._returns)
        historical_vol = math.sqrt(sum(r ** 2 for r in all_returns) / len(all_returns))

        if historical_vol <= 0:
            return None

        vol_ratio = recent_vol / historical_vol

        # 如果近期波动率是历史的3倍以上
        if vol_ratio > 3.0:
            return Anomaly(
                anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                severity=self._get_severity(vol_ratio),
                symbol=self._symbol,
                description=f"波动率异常飙升，是历史均值的 {vol_ratio:.1f} 倍",
                value=recent_vol,
                expected_range=(historical_vol * 0.5, historical_vol * 2),
                metadata={"vol_ratio": vol_ratio, "recent_vol": recent_vol, "historical_vol": historical_vol},
            )

        return None

    def _detect_volume_anomaly(self) -> Optional[Anomaly]:
        """检测成交量异常"""
        if len(self._volumes) < self._config["min_data_points"]:
            return None

        volumes = list(self._volumes)
        current_volume = volumes[-1]

        # 计算成交量均值（排除当前值）
        historical_volumes = volumes[:-1]
        mean_volume = sum(historical_volumes) / len(historical_volumes)

        if mean_volume <= 0:
            return None

        volume_ratio = current_volume / mean_volume

        # 如果成交量是均值的5倍以上
        if volume_ratio > 5.0:
            return Anomaly(
                anomaly_type=AnomalyType.VOLUME_ANOMALY,
                severity=self._get_severity(volume_ratio),
                symbol=self._symbol,
                description=f"成交量异常，是历史均值的 {volume_ratio:.1f} 倍",
                value=current_volume,
                expected_range=(mean_volume * 0.5, mean_volume * 3),
                metadata={"volume_ratio": volume_ratio, "mean_volume": mean_volume},
            )

        return None

    def _detect_trend_reversal(self) -> Optional[Anomaly]:
        """检测趋势反转"""
        if len(self._prices) < 50:
            return None

        prices = list(self._prices)

        # 计算短期和长期趋势
        short_window = 10
        long_window = 50

        short_ma = sum(prices[-short_window:]) / short_window
        long_ma = sum(prices[-long_window:]) / long_window
        prev_short_ma = sum(prices[-(short_window + 5):-5]) / short_window
        prev_long_ma = sum(prices[-(long_window + 5):-5]) / long_window

        # 检测交叉
        was_uptrend = prev_short_ma > prev_long_ma
        is_uptrend = short_ma > long_ma

        if was_uptrend != is_uptrend:
            direction = "下跌" if is_uptrend else "上涨"
            return Anomaly(
                anomaly_type=AnomalyType.TREND_REVERSAL,
                severity=Severity.HIGH,
                symbol=self._symbol,
                description=f"检测到趋势反转信号，可能转为{direction}趋势",
                value=short_ma,
                expected_range=(long_ma * 0.99, long_ma * 1.01),
                metadata={
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "was_uptrend": was_uptrend,
                },
            )

        return None

    def _detect_gap(self) -> Optional[Anomaly]:
        """检测价格缺口"""
        if len(self._prices) < 2:
            return None

        prices = list(self._prices)
        current = prices[-1]
        previous = prices[-2]

        if previous <= 0:
            return None

        gap_pct = abs(current - previous) / previous * 100

        # 如果价格缺口超过1%
        if gap_pct > 1.0:
            direction = "向上" if current > previous else "向下"
            return Anomaly(
                anomaly_type=AnomalyType.GAP_DETECTION,
                severity=self._get_severity(gap_pct),
                symbol=self._symbol,
                description=f"检测到{direction}价格缺口 {gap_pct:.2f}%",
                value=current,
                expected_range=(previous * 0.99, previous * 1.01),
                metadata={"gap_pct": gap_pct, "previous": previous},
            )

        return None

    def _update_stats_cache(self):
        """更新统计缓存"""
        if self._cache_valid:
            return

        prices = list(self._prices)

        if not prices:
            return

        # 计算均值和标准差
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(variance)

        self._stats_cache = {
            "price_mean": mean,
            "price_std": std,
            "price_min": min(prices),
            "price_max": max(prices),
            "data_points": len(prices),
        }

        self._cache_valid = True

    def _get_severity(self, value: float) -> Severity:
        """根据值确定严重程度"""
        if value > 5:
            return Severity.CRITICAL
        elif value > 4:
            return Severity.HIGH
        elif value > 3:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            self._update_stats_cache()
            return self._stats_cache.copy()

    def reset(self):
        """重置检测器"""
        with self._lock:
            self._prices.clear()
            self._volumes.clear()
            self._returns.clear()
            self._stats_cache.clear()
            self._cache_valid = False


# 全局异常检测器实例
_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """获取全局异常检测器实例"""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector