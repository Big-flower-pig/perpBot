"""
PerpBot 仓位管理模块

提供仓位管理功能：
- 获取当前持仓
- 计算仓位价值
- 计算盈亏
- 复利模式管理
- 本金追踪
"""

import os
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.helpers import safe_float, calculate_pnl, smart_price_format


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    unrealized_pnl: float = 0
    leverage: float = 10
    margin_mode: str = "isolated"
    contract_size: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    @property
    def position_value(self) -> float:
        """仓位价值 = 合约张数 × 入场价 × 合约乘数"""
        return self.size * self.entry_price * self.contract_size

    @property
    def margin_used(self) -> float:
        """已用保证金"""
        return self.position_value / self.leverage

    @property
    def pnl_percent(self) -> float:
        """盈亏百分比（相对保证金）"""
        if self.margin_used > 0:
            return (self.unrealized_pnl / self.margin_used) * 100
        return 0


@dataclass
class CapitalTracking:
    """本金追踪数据"""
    initial_capital: float
    current_capital: float
    total_pnl: float
    total_trades: int
    win_trades: int
    loss_trades: int
    last_update: datetime
    trade_history: List[Dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return (self.win_trades / self.total_trades) * 100

    @property
    def roi(self) -> float:
        """投资回报率"""
        if self.initial_capital > 0:
            return ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        return 0


class PositionManager:
    """仓位管理器

    提供仓位管理功能

    Example:
        position_manager = PositionManager()

        # 获取当前仓位
        position = position_manager.get_position()

        # 计算仓位大小
        size = position_manager.calculate_size(capital, price)

        # 更新本金追踪
        position_manager.update_capital(pnl=10.5)
    """

    _instance = None
    _lock = threading.Lock()

    # 本金追踪文件路径
    TRACKING_FILE = "data/tracking/capital_tracking.json"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._logger = get_logger("position")
        self._tracking: Optional[CapitalTracking] = None
        self._initialized = True

    def get_position(self, exchange) -> Optional[Position]:
        """获取当前持仓

        Args:
            exchange: 交易所实例

        Returns:
            Position 对象或 None
        """
        try:
            symbol = get_config("trading.symbol")
            positions = exchange.fetch_positions([symbol])

            for pos in positions:
                if pos["symbol"] == symbol:
                    contracts = safe_float(pos.get("contracts"))
                    if contracts and contracts > 0:
                        return Position(
                            symbol=symbol,
                            side=pos.get("side"),
                            size=contracts,
                            entry_price=safe_float(pos.get("entryPrice")),
                            unrealized_pnl=safe_float(pos.get("unrealizedPnl")),
                            leverage=safe_float(pos.get("leverage", get_config("trading.leverage"))),
                            margin_mode=pos.get("mgnMode", get_config("trading.margin_mode")),
                            contract_size=exchange.contract_size,
                        )

            return None

        except Exception as e:
            self._logger.error(f"获取持仓失败: {e}")
            return None

    def calculate_size(
        self,
        capital: float,
        price: float,
        confidence: str = "MEDIUM",
        leverage: int = None,
        contract_size: float = None,
    ) -> float:
        """计算仓位大小

        Args:
            capital: 本金
            price: 当前价格
            confidence: 信心程度
            leverage: 杠杆倍数
            contract_size: 合约乘数

        Returns:
            合约张数
        """
        leverage = leverage or get_config("trading.leverage", 10)
        contract_size = contract_size or 1.0

        position_config = get_config("position", {})
        mode = position_config.get("mode", "compound")

        # 获取基础本金
        if mode == "compound":
            # 复利模式：使用追踪的本金
            tracking = self.get_tracking()
            base_capital = tracking.current_capital if tracking else capital
        else:
            # 固定模式：使用配置的固定金额
            base_capital = position_config.get("base_usdt_amount", capital)

        # 信心倍数
        confidence_multipliers = position_config.get("confidence_multipliers", {})
        multiplier = confidence_multipliers.get(confidence.lower(), 1.0)

        # 调整本金
        adjusted_capital = base_capital * multiplier

        # 确保不超过实际资金
        adjusted_capital = min(adjusted_capital, capital)

        # 计算合约张数
        # 公式：合约张数 = (本金 × 杠杆) ÷ (价格 × 合约乘数)
        size = (adjusted_capital * leverage) / (price * contract_size)

        # 保留2位小数
        size = round(size, 2)

        # 确保最小交易量
        min_amount = 0.01
        if size < min_amount:
            size = min_amount

        self._logger.debug(
            f"仓位计算: 本金={adjusted_capital:.2f}, 杠杆={leverage}x, "
            f"价格={price:.4f}, 结果={size}张"
        )

        return size

    def get_tracking(self) -> Optional[CapitalTracking]:
        """获取本金追踪数据

        Returns:
            CapitalTracking 对象或 None
        """
        if self._tracking:
            return self._tracking

        try:
            tracking_path = Path(self.TRACKING_FILE)

            if not tracking_path.exists():
                return None

            with open(tracking_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._tracking = CapitalTracking(
                initial_capital=data.get("initial_capital", 0),
                current_capital=data.get("current_capital", 0),
                total_pnl=data.get("total_pnl", 0),
                total_trades=data.get("total_trades", 0),
                win_trades=data.get("win_trades", 0),
                loss_trades=data.get("loss_trades", 0),
                last_update=datetime.fromisoformat(data.get("last_update", datetime.now().isoformat())),
                trade_history=data.get("trade_history", []),
            )

            return self._tracking

        except Exception as e:
            self._logger.error(f"加载本金追踪数据失败: {e}")
            return None

    def init_tracking(self, initial_capital: float) -> CapitalTracking:
        """初始化本金追踪

        Args:
            initial_capital: 初始本金

        Returns:
            CapitalTracking 对象
        """
        self._tracking = CapitalTracking(
            initial_capital=initial_capital,
            current_capital=initial_capital,
            total_pnl=0,
            total_trades=0,
            win_trades=0,
            loss_trades=0,
            last_update=datetime.now(),
            trade_history=[],
        )

        self._save_tracking()
        self._logger.info(f"初始化本金追踪: 初始本金 {initial_capital} USDT")

        return self._tracking

    def update_tracking(
        self,
        pnl: float,
        trade_info: Optional[Dict] = None,
    ) -> CapitalTracking:
        """更新本金追踪

        Args:
            pnl: 盈亏金额
            trade_info: 交易信息

        Returns:
            更新后的 CapitalTracking 对象
        """
        # 如果没有追踪数据，初始化
        if not self._tracking:
            initial_capital = get_config("position.initial_capital", 100)
            self._tracking = self.init_tracking(initial_capital)

        # 更新统计
        self._tracking.total_pnl += pnl
        self._tracking.current_capital = self._tracking.initial_capital + self._tracking.total_pnl
        self._tracking.total_trades += 1
        self._tracking.last_update = datetime.now()

        if pnl >= 0:
            self._tracking.win_trades += 1
        else:
            self._tracking.loss_trades += 1

        # 记录交易历史
        if trade_info:
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "pnl": pnl,
                **trade_info,
            }
            self._tracking.trade_history.append(trade_record)

            # 限制历史长度
            if len(self._tracking.trade_history) > 100:
                self._tracking.trade_history = self._tracking.trade_history[-50:]

        # 保存
        self._save_tracking()

        # 记录日志
        self._logger.info(
            f"本金更新: 初始{self._tracking.initial_capital:.2f} + "
            f"累计盈亏{self._tracking.total_pnl:.2f} = "
            f"当前{self._tracking.current_capital:.2f} USDT"
        )

        return self._tracking

    def _save_tracking(self):
        """保存本金追踪数据"""
        if not self._tracking:
            return

        try:
            tracking_path = Path(self.TRACKING_FILE)
            tracking_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "initial_capital": self._tracking.initial_capital,
                "current_capital": self._tracking.current_capital,
                "total_pnl": self._tracking.total_pnl,
                "total_trades": self._tracking.total_trades,
                "win_trades": self._tracking.win_trades,
                "loss_trades": self._tracking.loss_trades,
                "last_update": self._tracking.last_update.isoformat(),
                "trade_history": self._tracking.trade_history[-50:],  # 只保存最近50条
            }

            with open(tracking_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self._logger.error(f"保存本金追踪数据失败: {e}")

    def get_trading_capital(self) -> float:
        """获取当前可用于交易的本金

        Returns:
            当前本金
        """
        tracking = self.get_tracking()

        if not tracking:
            # 没有追踪数据，使用配置的初始本金
            initial_capital = get_config("position.initial_capital", 100)
            return initial_capital

        return tracking.current_capital

    def calculate_pnl_for_position(
        self,
        position: Position,
        current_price: float,
    ) -> Dict[str, float]:
        """计算持仓盈亏

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            盈亏信息字典
        """
        return calculate_pnl(
            entry_price=position.entry_price,
            exit_price=current_price,
            size=position.size,
            side=position.side,
            leverage=position.leverage,
            contract_size=position.contract_size,
        )

    def get_position_summary(self, position: Position, current_price: float) -> Dict[str, Any]:
        """获取持仓摘要

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            持仓摘要字典
        """
        pnl_info = self.calculate_pnl_for_position(position, current_price)

        return {
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "position_value": position.position_value,
            "margin_used": position.margin_used,
            "unrealized_pnl": position.unrealized_pnl,
            "pnl_usdt": pnl_info["pnl_usdt"],
            "pnl_pct": pnl_info["pnl_pct"],
            "roi": pnl_info["roi"],
            "leverage": position.leverage,
            "margin_mode": position.margin_mode,
        }

    def reset_tracking(self, initial_capital: float = None):
        """重置本金追踪

        Args:
            initial_capital: 新的初始本金
        """
        initial_capital = initial_capital or get_config("position.initial_capital", 100)
        self._tracking = None
        self.init_tracking(initial_capital)
        self._logger.warning(f"本金追踪已重置: {initial_capital} USDT")


# 全局仓位管理器实例
_position_manager: Optional[PositionManager] = None


def get_position_manager() -> PositionManager:
    """获取全局仓位管理器实例"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager