"""
PerpBot 核心模块

提供核心交易功能：
- ExchangeManager: 交易所管理
- StrategyEngine: 策略引擎
- RiskManager: 风险管理
- PositionManager: 仓位管理
"""

from core.exchange import ExchangeManager, get_exchange_manager
from core.strategy import StrategyEngine, SignalResult, TrendAnalysis
from core.risk_manager import RiskManager, get_risk_manager, RiskLevel, RiskAction
from core.position import PositionManager, get_position_manager, Position, CapitalTracking

__all__ = [
    # Exchange
    "ExchangeManager",
    "get_exchange_manager",
    # Strategy
    "StrategyEngine",
    "SignalResult",
    "TrendAnalysis",
    # Risk
    "RiskManager",
    "get_risk_manager",
    "RiskLevel",
    "RiskAction",
    # Position
    "PositionManager",
    "get_position_manager",
    "Position",
    "CapitalTracking",
]