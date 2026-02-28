"""
PerpBot 价格监控模块

提供实时价格监控功能：
- 价格变动追踪
- 价格警报
- 成交量监控
- 资金费率监控
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.helpers import safe_float


class AlertType(Enum):
    """警报类型"""
    PRICE_CHANGE = "price_change"
    VOLUME_SPIKE = "volume_spike"
    FUNDING_RATE = "funding_rate"
    RAPID_MOVE = "rapid_move"
    BREAKOUT = "breakout"


@dataclass
class PriceData:
    """价格数据"""
    price: float
    timestamp: datetime
    volume: float = 0
    funding_rate: Optional[float] = None


@dataclass
class PriceAlert:
    """价格警报"""
    alert_type: AlertType
    symbol: str
    current_price: float
    threshold_value: float
    actual_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "symbol": self.symbol,
            "current_price": self.current_price,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class PriceMonitor:
    """价格监控器

    监控价格变动并触发警报

    Example:
        monitor = PriceMonitor()

        # 更新价格
        monitor.update(price=95000, volume=1000000)

        # 检查警报
        alerts = monitor.check_alerts()

        # 注册回调
        monitor.on_alert(callback)
    """

    def __init__(self):
        self._logger = get_logger("monitor")
        self._symbol = get_config("trading.symbol", "BTC/USDT:USDT")

        # 价格历史 (保留最近1000个数据点)
        self._price_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

        # 警报阈值配置
        self._thresholds = {
            "price_change_pct": get_config("monitoring.alerts.price_change_pct", 2.0),
            "volume_spike_multiplier": get_config("monitoring.alerts.volume_spike_multiplier", 3.0),
            "funding_rate_high": get_config("monitoring.alerts.funding_rate_high", 0.001),
            "rapid_move_seconds": get_config("monitoring.alerts.rapid_move_seconds", 60),
            "rapid_move_pct": get_config("monitoring.alerts.rapid_move_pct", 1.0),
        }

        # 警报回调
        self._alert_callbacks: List[Callable[[PriceAlert], None]] = []

        # 上次检查时间
        self._last_check = datetime.now()

        # 成交量基线
        self._volume_baseline: Optional[float] = None

    def update(
        self,
        price: float,
        volume: float = 0,
        funding_rate: Optional[float] = None,
    ) -> List[PriceAlert]:
        """更新价格数据并检查警报

        Args:
            price: 当前价格
            volume: 成交量
            funding_rate: 资金费率

        Returns:
            触发的警报列表
        """
        with self._lock:
            # 记录价格数据
            price_data = PriceData(
                price=price,
                timestamp=datetime.now(),
                volume=volume,
                funding_rate=funding_rate,
            )
            self._price_history.append(price_data)

            # 更新成交量基线
            self._update_volume_baseline(volume)

        # 检查警报
        return self.check_alerts()

    def check_alerts(self) -> List[PriceAlert]:
        """检查所有警报条件

        Returns:
            触发的警报列表
        """
        alerts = []

        with self._lock:
            if len(self._price_history) < 2:
                return alerts

            current = self._price_history[-1]
            previous = self._price_history[-2]

        # 检查价格变动
        price_alert = self._check_price_change(current)
        if price_alert:
            alerts.append(price_alert)

        # 检查快速移动
        rapid_alert = self._check_rapid_move(current)
        if rapid_alert:
            alerts.append(rapid_alert)

        # 检查成交量异常
        volume_alert = self._check_volume_spike(current)
        if volume_alert:
            alerts.append(volume_alert)

        # 检查资金费率
        funding_alert = self._check_funding_rate(current)
        if funding_alert:
            alerts.append(funding_alert)

        # 触发回调
        for alert in alerts:
            self._trigger_callbacks(alert)

        return alerts

    def _check_price_change(self, current: PriceData) -> Optional[PriceAlert]:
        """检查价格变动警报"""
        threshold = self._thresholds["price_change_pct"]

        # 计算相对24小时前的变化
        with self._lock:
            if len(self._price_history) < 10:
                return None

            # 获取最早的价格作为基准
            oldest = self._price_history[0]
            if oldest.price <= 0:
                return None

            change_pct = ((current.price - oldest.price) / oldest.price) * 100

        if abs(change_pct) >= threshold:
            direction = "上涨" if change_pct > 0 else "下跌"
            return PriceAlert(
                alert_type=AlertType.PRICE_CHANGE,
                symbol=self._symbol,
                current_price=current.price,
                threshold_value=threshold,
                actual_value=change_pct,
                message=f"{self._symbol} {direction} {abs(change_pct):.2f}%，超过阈值 {threshold}%",
            )

        return None

    def _check_rapid_move(self, current: PriceData) -> Optional[PriceAlert]:
        """检查快速移动警报"""
        threshold_pct = self._thresholds["rapid_move_pct"]
        threshold_seconds = self._thresholds["rapid_move_seconds"]

        with self._lock:
            # 查找指定时间前的价格
            cutoff_time = datetime.now().timestamp() - threshold_seconds

            for data in reversed(self._price_history):
                if data.timestamp.timestamp() < cutoff_time:
                    if data.price <= 0:
                        continue

                    change_pct = ((current.price - data.price) / data.price) * 100

                    if abs(change_pct) >= threshold_pct:
                        direction = "急涨" if change_pct > 0 else "急跌"
                        return PriceAlert(
                            alert_type=AlertType.RAPID_MOVE,
                            symbol=self._symbol,
                            current_price=current.price,
                            threshold_value=threshold_pct,
                            actual_value=change_pct,
                            message=f"{self._symbol} {threshold_seconds}秒内{direction} {abs(change_pct):.2f}%",
                        )
                    break

        return None

    def _check_volume_spike(self, current: PriceData) -> Optional[PriceAlert]:
        """检查成交量异常"""
        if not self._volume_baseline or self._volume_baseline <= 0:
            return None

        multiplier = self._thresholds["volume_spike_multiplier"]

        if current.volume > self._volume_baseline * multiplier:
            return PriceAlert(
                alert_type=AlertType.VOLUME_SPIKE,
                symbol=self._symbol,
                current_price=current.price,
                threshold_value=self._volume_baseline * multiplier,
                actual_value=current.volume,
                message=f"{self._symbol} 成交量异常: {current.volume:,.0f}，"
                        f"是基线的 {current.volume / self._volume_baseline:.1f} 倍",
            )

        return None

    def _check_funding_rate(self, current: PriceData) -> Optional[PriceAlert]:
        """检查资金费率警报"""
        if current.funding_rate is None:
            return None

        threshold = self._thresholds["funding_rate_high"]

        if abs(current.funding_rate) >= threshold:
            direction = "多头付费" if current.funding_rate > 0 else "空头付费"
            return PriceAlert(
                alert_type=AlertType.FUNDING_RATE,
                symbol=self._symbol,
                current_price=current.price,
                threshold_value=threshold,
                actual_value=current.funding_rate,
                message=f"{self._symbol} 资金费率异常: {current.funding_rate * 100:.4f}% ({direction})",
            )

        return None

    def _update_volume_baseline(self, volume: float):
        """更新成交量基线"""
        if volume <= 0:
            return

        if self._volume_baseline is None:
            self._volume_baseline = volume
        else:
            # 使用指数移动平均更新基线
            alpha = 0.1
            self._volume_baseline = alpha * volume + (1 - alpha) * self._volume_baseline

    def on_alert(self, callback: Callable[[PriceAlert], None]):
        """注册警报回调

        Args:
            callback: 回调函数
        """
        self._alert_callbacks.append(callback)

    def _trigger_callbacks(self, alert: PriceAlert):
        """触发所有回调"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._logger.error(f"警报回调执行失败: {e}")

    def get_price_stats(self) -> Dict[str, Any]:
        """获取价格统计信息"""
        with self._lock:
            if len(self._price_history) < 2:
                return {}

            prices = [p.price for p in self._price_history]
            volumes = [p.volume for p in self._price_history if p.volume > 0]

            return {
                "current_price": prices[-1],
                "high": max(prices),
                "low": min(prices),
                "avg_price": sum(prices) / len(prices),
                "price_count": len(prices),
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
                "volume_baseline": self._volume_baseline,
            }

    def get_recent_prices(self, count: int = 100) -> List[PriceData]:
        """获取最近的价格数据"""
        with self._lock:
            return list(self._price_history)[-count:]


# 全局监控器实例
_monitor: Optional[PriceMonitor] = None


def get_price_monitor() -> PriceMonitor:
    """获取全局价格监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = PriceMonitor()
    return _monitor