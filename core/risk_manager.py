"""
PerpBot 风险管理模块

提供完善的风险管理功能：
- 止损止盈计算
- 仓位大小计算 (含凯利公式)
- 最大回撤控制
- VaR 计算
- 交易频率限制
- 日内亏损限制
- 夏普比率计算
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import threading
import statistics

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.helpers import safe_float, calculate_pnl, calculate_liquidation_price


class RiskLevel(Enum):
    """风险等级"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAction(Enum):
    """风险动作"""

    ALLOW = "ALLOW"  # 允许交易
    WARN = "WARN"  # 警告但允许
    REDUCE = "REDUCE"  # 减少仓位
    BLOCK = "BLOCK"  # 阻止交易


@dataclass
class RiskAssessment:
    """风险评估结果"""

    action: RiskAction
    level: RiskLevel
    reason: str
    suggested_size: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        """交易是否被批准"""
        return self.action in [RiskAction.ALLOW, RiskAction.WARN]

    @property
    def message(self) -> str:
        """风险消息（兼容旧代码）"""
        return self.reason


@dataclass
class DailyStats:
    """日内统计"""

    date: date
    trades: int = 0
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_capital: float = 0.0


@dataclass
class PerformanceMetrics:
    """性能指标"""

    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    kelly_fraction: float = 0.0  # 凯利比例


class RiskManager:
    """风险管理器

    提供完善的风险管理功能

    Example:
        risk_manager = RiskManager()

        # 评估交易风险
        assessment = risk_manager.assess_trade(signal, position, capital)
        if assessment.action == RiskAction.BLOCK:
            print("交易被阻止:", assessment.reason)

        # 计算仓位大小
        size = risk_manager.calculate_position_size(capital, price, signal)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._logger = get_logger("risk")
        self._daily_stats: Dict[date, DailyStats] = {}
        self._trade_history: List[Dict] = []
        self._capital_history: List[float] = []
        self._initialized = True

    def assess_trade(
        self,
        signal: str = None,
        confidence: str = "MEDIUM",
        position: Optional[Dict] = None,
        capital: float = 0,
        current_price: float = 0,
        symbol: str = None,
        side: str = None,
    ) -> RiskAssessment:
        """评估交易风险

        Args:
            signal: 交易信号 (BUY/SELL/HOLD) - 兼容旧参数
            confidence: 信心程度 (HIGH/MEDIUM/LOW)
            position: 当前持仓
            capital: 当前资金
            current_price: 当前价格
            symbol: 交易对 - 新参数
            side: 交易方向 (long/short) - 新参数

        Returns:
            RiskAssessment 对象
        """
        # 兼容新旧参数：如果传入了 side，则转换为 signal
        if signal is None and side is not None:
            signal = "BUY" if side == "long" else "SELL"

        warnings = []
        details = {}

        # 记录交易对信息
        if symbol:
            details["symbol"] = symbol

        # HOLD 信号或无信号直接允许
        if signal is None or signal == "HOLD":
            return RiskAssessment(
                action=RiskAction.ALLOW,
                level=RiskLevel.LOW,
                reason="HOLD信号无需评估",
            )

        # 1. 检查信心程度
        if confidence == "LOW":
            skip_low = get_config("risk.skip_low_confidence", True)
            if skip_low:
                return RiskAssessment(
                    action=RiskAction.BLOCK,
                    level=RiskLevel.HIGH,
                    reason="低信心信号被阻止",
                    warnings=["信心程度过低"],
                )
            warnings.append("低信心信号，建议谨慎")

        # 2. 检查日内交易次数
        today = date.today()
        daily_stats = self._daily_stats.get(today, DailyStats(date=today))
        max_trades = get_config("risk.max_trades_per_day", 20)

        if daily_stats.trades >= max_trades:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                level=RiskLevel.HIGH,
                reason=f"已达日内最大交易次数 {max_trades}",
                warnings=[f"今日已交易 {daily_stats.trades} 次"],
            )

        details["trades_today"] = daily_stats.trades

        # 3. 检查日内亏损
        max_daily_loss = get_config("risk.max_daily_loss", 10.0)
        if capital > 0:
            daily_loss_pct = (daily_stats.pnl / capital) * 100
            details["daily_loss_pct"] = daily_loss_pct

            if daily_loss_pct <= -max_daily_loss:
                return RiskAssessment(
                    action=RiskAction.BLOCK,
                    level=RiskLevel.CRITICAL,
                    reason=f"已达日内最大亏损 {max_daily_loss}%",
                    warnings=[f"今日亏损: {daily_loss_pct:.2f}%"],
                )
            elif daily_loss_pct <= -max_daily_loss * 0.7:
                warnings.append(f"接近日内亏损限制: {daily_loss_pct:.2f}%")

        # 4. 检查持仓情况
        if position:
            # 已有持仓时检查是否反向操作
            current_side = position.get("side")
            new_side = "long" if signal == "BUY" else "short"

            if current_side == new_side:
                warnings.append("同方向加仓")
            else:
                warnings.append("反向操作，将先平仓")

            # 检查未实现盈亏
            unrealized_pnl = position.get("unrealized_pnl", 0)
            if unrealized_pnl < 0:
                warnings.append(f"当前持仓亏损: {unrealized_pnl:.2f} USDT")

        # 5. 检查连续亏损
        consecutive_losses = self._get_consecutive_losses()
        if consecutive_losses >= 3:
            warnings.append(f"连续亏损 {consecutive_losses} 次，建议降低仓位")

        # 6. 计算建议仓位
        suggested_size = self.calculate_position_size(
            capital=capital,
            price=current_price,
            confidence=confidence,
        )

        # 确定风险等级和动作
        if len(warnings) >= 3:
            level = RiskLevel.HIGH
            action = RiskAction.WARN
        elif len(warnings) >= 1:
            level = RiskLevel.MEDIUM
            action = RiskAction.ALLOW
        else:
            level = RiskLevel.LOW
            action = RiskAction.ALLOW

        # 记录风险评估日志
        if hasattr(self._logger, "risk"):
            self._logger.risk(
                event="TRADE_ASSESSMENT",
                level=level.value,
                details={
                    "signal": signal,
                    "confidence": confidence,
                    "action": action.value,
                    "warnings": warnings,
                },
            )
        else:
            self._logger.info(
                f"[RISK] TRADE_ASSESSMENT - level={level.value}, "
                f"signal={signal}, action={action.value}"
            )

        return RiskAssessment(
            action=action,
            level=level,
            reason="风险评估通过",
            suggested_size=suggested_size,
            warnings=warnings,
            details=details,
        )

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        confidence: str = "MEDIUM",
        leverage: int = None,
        contract_size: float = 1.0,
        use_kelly: bool = True,
    ) -> float:
        """计算仓位大小 (优化版: 支持凯利公式)

        核心改进:
        1. 基于历史表现计算凯利比例
        2. 使用半凯利以降低风险
        3. 结合信心程度调整

        Args:
            capital: 本金
            price: 当前价格
            confidence: 信心程度
            leverage: 杠杆倍数
            contract_size: 合约乘数
            use_kelly: 是否使用凯利公式

        Returns:
            建议仓位大小（合约张数）
        """
        if capital <= 0 or price <= 0:
            return 0

        # 获取配置
        leverage = leverage or get_config("trading.leverage", 10)
        position_config = get_config("position", {})
        mode = position_config.get("mode", "compound")

        # 获取基础本金
        if mode == "compound":
            base_capital = position_config.get("initial_capital", capital)
        else:
            base_capital = position_config.get("base_usdt_amount", capital)

        # ===== 核心改进: 凯利公式计算 =====
        kelly_fraction = 0.25  # 默认使用25%仓位

        if use_kelly and len(self._trade_history) >= 10:
            # 有足够历史数据时，计算凯利比例
            kelly_fraction = self._calculate_kelly_fraction()

        # 信心倍数调整
        confidence_multipliers = position_config.get("confidence_multipliers", {})
        confidence_mult = confidence_multipliers.get(confidence.lower(), 1.0)

        # 综合调整后的仓位比例
        position_fraction = kelly_fraction * confidence_mult

        # 最大仓位限制 (不超过50%)
        max_fraction = position_config.get("max_position_ratio", 0.5)
        position_fraction = min(position_fraction, max_fraction)

        # 计算调整后的本金
        adjusted_capital = base_capital * position_fraction

        # 确保不超过实际资金
        adjusted_capital = min(adjusted_capital, capital)

        # 计算仓位大小
        # 公式：合约张数 = (本金 × 杠杆) ÷ (价格 × 合约乘数)
        size = (adjusted_capital * leverage) / (price * contract_size)

        # 保留2位小数
        size = round(size, 2)

        # 确保最小交易量
        min_amount = get_config("exchange.min_amount", 0.01)
        if size < min_amount:
            size = min_amount

        self._logger.debug(
            f"仓位计算: 本金={adjusted_capital:.2f}, 凯利比例={kelly_fraction:.1%}, "
            f"信心倍数={confidence_mult}, 杠杆={leverage}x, 结果={size}张"
        )

        return size

    def _calculate_kelly_fraction(
        self,
        max_fraction: float = 0.5,
        use_half_kelly: bool = True,
    ) -> float:
        """计算凯利公式比例

        凯利公式: f* = p - (1-p) / (b)
        其中:
        - p = 胜率
        - b = 平均盈利 / 平均亏损 (盈亏比)

        Args:
            max_fraction: 最大仓位比例
            use_half_kelly: 是否使用半凯利 (更保守)

        Returns:
            凯利比例 (0 ~ max_fraction)
        """
        if len(self._trade_history) < 10:
            # 历史数据不足，使用默认值
            return 0.25

        # 计算胜率和盈亏比
        pnls = [t.get("pnl", 0) for t in self._trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses:
            return 0.25

        # 胜率
        win_rate = len(wins) / len(pnls)

        # 盈亏比
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss == 0:
            return max_fraction

        win_loss_ratio = avg_win / avg_loss

        # 凯利公式
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # 凯利值可能为负，表示不应该下注
        if kelly <= 0:
            return 0.1  # 最小仓位

        # 使用半凯利以降低风险
        if use_half_kelly:
            kelly = kelly * 0.5

        # 限制最大比例
        kelly = min(kelly, max_fraction)

        self._logger.debug(
            f"凯利计算: 胜率={win_rate:.1%}, 盈亏比={win_loss_ratio:.2f}, "
            f"凯利={kelly:.1%}"
        )

        return kelly

    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 (优化版)

        Returns:
            PerformanceMetrics 对象
        """
        if not self._trade_history:
            return PerformanceMetrics()

        trades = self._trade_history
        pnls = [t.get("pnl", 0) for t in trades]

        total_trades = len(trades)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        profit_factor = (
            abs(sum(wins) / sum(losses))
            if losses and sum(losses) != 0
            else float("inf")
        )

        # 计算夏普比率
        if len(pnls) > 1:
            avg_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = (avg_pnl / std_pnl * (252**0.5)) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # 计算最大回撤
        cumulative = []
        running_sum = 0
        for pnl in pnls:
            running_sum += pnl
            cumulative.append(running_sum)

        max_dd = 0
        peak = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # 计算凯利比例
        kelly = self._calculate_kelly_fraction()

        return PerformanceMetrics(
            total_trades=total_trades,
            win_trades=len(wins),
            loss_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd * 100,
            kelly_fraction=kelly,
        )

    def calculate_var(
        self,
        capital: float,
        confidence_level: float = 0.95,
        holding_period_days: int = 1,
    ) -> Dict[str, float]:
        """计算 VaR (风险价值)

        使用历史模拟法计算 VaR，估计在给定置信水平下的最大可能损失。

        Args:
            capital: 当前资金
            confidence_level: 置信水平 (0.95 = 95%)
            holding_period_days: 持有期 (天)

        Returns:
            包含 VaR 信息的字典
        """
        if len(self._trade_history) < 20:
            # 历史数据不足
            return {
                "var": 0,
                "var_percent": 0,
                "confidence_level": confidence_level,
                "method": "insufficient_data",
            }

        # 获取历史收益率
        pnls = [t.get("pnl", 0) for t in self._trade_history]

        # 计算收益率 (假设每笔交易使用相同本金)
        if capital <= 0:
            return {"var": 0, "var_percent": 0}

        # 使用百分位数计算 VaR
        var_threshold = statistics.quantiles(pnls, n=100)[
            int((1 - confidence_level) * 100)
        ]

        # 调整持有期 (按平方根法则)
        var_adjusted = abs(var_threshold) * (holding_period_days**0.5)
        var_percent = (var_adjusted / capital) * 100

        return {
            "var": var_adjusted,
            "var_percent": var_percent,
            "confidence_level": confidence_level,
            "holding_period_days": holding_period_days,
            "method": "historical_simulation",
        }

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        side: str,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ) -> Tuple[float, float]:
        """计算止损止盈价格

        Args:
            entry_price: 入场价格
            side: 方向 (long/short)
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比

        Returns:
            (止损价格, 止盈价格)
        """
        stop_loss_pct = stop_loss_pct or get_config("risk.stop_loss_percent", 3.0)
        take_profit_pct = take_profit_pct or get_config("risk.take_profit_percent", 5.0)

        if side == "long":
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            take_profit = entry_price * (1 + take_profit_pct / 100)
        else:  # short
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            take_profit = entry_price * (1 - take_profit_pct / 100)

        return stop_loss, take_profit

    def check_position_risk(
        self,
        position,  # 支持 Dict 或 Position dataclass
        current_price: float,
    ) -> RiskAssessment:
        """检查持仓风险

        Args:
            position: 持仓信息（字典或 Position dataclass）
            current_price: 当前价格

        Returns:
            RiskAssessment 对象
        """
        # 支持 dataclass 和字典两种类型
        if hasattr(position, "entry_price"):
            # Position dataclass
            entry_price = position.entry_price
            side = position.side
            unrealized_pnl = position.unrealized_pnl
            size = position.size
            leverage = getattr(position, "leverage", 10)
        else:
            # Dict
            entry_price = position.get("entry_price", 0)
            side = position.get("side", "long")
            unrealized_pnl = position.get("unrealized_pnl", 0)
            size = position.get("size", 0)
            leverage = position.get("leverage", 10)

        if entry_price <= 0:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                level=RiskLevel.LOW,
                reason="无有效持仓",
            )

        # 计算盈亏百分比
        pnl_result = calculate_pnl(
            entry_price=entry_price,
            exit_price=current_price,
            size=size,
            side=side,
        )
        pnl_pct = pnl_result["pnl_pct"]

        # 获取风控阈值
        stop_loss_pct = get_config("risk.stop_loss_percent", 3.0)
        take_profit_pct = get_config("risk.take_profit_percent", 5.0)
        max_drawdown = get_config("risk.max_drawdown", 20.0)

        warnings = []
        action = RiskAction.ALLOW
        level = RiskLevel.LOW

        # 检查止损
        if pnl_pct <= -stop_loss_pct:
            action = RiskAction.BLOCK
            level = RiskLevel.CRITICAL
            warnings.append(f"触发止损: 亏损 {abs(pnl_pct):.2f}%")
        elif pnl_pct <= -stop_loss_pct * 0.8:
            level = RiskLevel.HIGH
            warnings.append(f"接近止损: 亏损 {abs(pnl_pct):.2f}%")

        # 检查止盈
        if pnl_pct >= take_profit_pct:
            level = RiskLevel.MEDIUM
            warnings.append(f"触发止盈: 盈利 {pnl_pct:.2f}%")

        # 检查最大回撤
        if pnl_pct <= -max_drawdown:
            action = RiskAction.BLOCK
            level = RiskLevel.CRITICAL
            warnings.append(f"触发最大回撤限制: {abs(pnl_pct):.2f}%")

        # 检查强平风险（leverage 已在上面提取）
        liquidation_price = calculate_liquidation_price(entry_price, leverage, side)
        price_distance = abs(current_price - liquidation_price) / current_price * 100

        if price_distance < 2:
            action = RiskAction.BLOCK
            level = RiskLevel.CRITICAL
            warnings.append(f"接近强平价: 距离 {price_distance:.2f}%")
        elif price_distance < 5:
            level = RiskLevel.HIGH
            warnings.append(f"强平风险: 距离 {price_distance:.2f}%")

        return RiskAssessment(
            action=action,
            level=level,
            reason="持仓风险检查完成",
            warnings=warnings,
            details={
                "pnl_pct": pnl_pct,
                "liquidation_price": liquidation_price,
                "price_distance": price_distance,
            },
        )

    def record_trade(
        self,
        signal: str,
        size: float,
        entry_price: float,
        exit_price: float = None,
        pnl: float = 0,
        side: str = None,
    ):
        """记录交易

        Args:
            signal: 交易信号
            size: 仓位大小
            entry_price: 入场价
            exit_price: 出场价
            pnl: 盈亏
            side: 方向
        """
        today = date.today()
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(date=today)

        stats = self._daily_stats[today]
        stats.trades += 1
        stats.pnl += pnl

        if pnl >= 0:
            stats.wins += 1
        else:
            stats.losses += 1

        # 记录交易历史
        trade = {
            "timestamp": datetime.now(),
            "signal": signal,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "side": side,
        }
        self._trade_history.append(trade)

        # 限制历史长度
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-500:]

    def _get_consecutive_losses(self) -> int:
        """获取连续亏损次数"""
        count = 0
        for trade in reversed(self._trade_history):
            if trade.get("pnl", 0) < 0:
                count += 1
            else:
                break
        return count

    def get_daily_stats(self, day: date = None) -> DailyStats:
        """获取日内统计"""
        day = day or date.today()
        return self._daily_stats.get(day, DailyStats(date=day))

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要 (用于日志和报告)"""
        metrics = self.get_performance_metrics()
        return {
            "total_trades": metrics.total_trades,
            "win_rate": f"{metrics.win_rate:.1f}%",
            "profit_factor": f"{metrics.profit_factor:.2f}",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{metrics.max_drawdown:.1f}%",
            "kelly_fraction": f"{metrics.kelly_fraction:.1%}",
            "avg_win": f"${metrics.avg_win:.2f}",
            "avg_loss": f"${metrics.avg_loss:.2f}",
        }

    def reset_daily_stats(self):
        """重置日内统计"""
        today = date.today()
        self._daily_stats[today] = DailyStats(date=today)


# Global risk manager instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get global risk manager instance"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
