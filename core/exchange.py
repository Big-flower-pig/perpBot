"""
PerpBot 交易所管理模块

封装交易所 API，提供统一的交易接口：
- 初始化交易所连接
- 验证 API 连接
- 设置杠杆和仓位模式
- 获取市场数据
- 执行订单
- 获取账户信息
- 自动重连机制
"""

import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import ccxt

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.decorators import retry_on_failure, rate_limit, log_execution_time
from utils.helpers import safe_float, smart_price_format


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"


class MarginMode(Enum):
    """仓位模式"""
    CROSS = "cross"
    ISOLATED = "isolated"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Ticker:
    """行情数据"""
    symbol: str
    bid: float  # 买一价
    ask: float  # 卖一价
    last: float  # 最新价
    high: float  # 24h最高
    low: float  # 24h最低
    volume: float  # 24h成交量
    timestamp: datetime

    @property
    def spread(self) -> float:
        """价差"""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """价差百分比"""
        return (self.spread / self.last) * 100 if self.last > 0 else 0


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # 'long' 或 'short'
    size: float  # 合约张数
    entry_price: float  # 入场价
    unrealized_pnl: float  # 未实现盈亏
    leverage: float  # 杠杆
    margin_mode: str  # 仓位模式
    liquidation_price: Optional[float] = None  # 强平价
    timestamp: Optional[datetime] = None

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    @property
    def position_value(self) -> float:
        """仓位价值"""
        return self.size * self.entry_price

    @property
    def pnl_percent(self) -> float:
        """盈亏百分比（相对入场价）"""
        if self.entry_price > 0:
            return (self.unrealized_pnl / (self.position_value / self.leverage)) * 100
        return 0


@dataclass
class OrderResult:
    """订单结果"""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    type: Optional[str] = None
    price: Optional[float] = None
    size: Optional[float] = None
    filled_size: Optional[float] = None
    average_price: Optional[float] = None
    fee: Optional[float] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


class ExchangeError(Exception):
    """交易所错误"""
    pass


class ConnectionError(ExchangeError):
    """连接错误"""
    pass


class OrderError(ExchangeError):
    """订单错误"""
    pass


class InsufficientFundsError(ExchangeError):
    """资金不足错误"""
    pass


class ExchangeManager:
    """交易所管理器

    封装交易所 API，提供统一的交易接口

    Example:
        exchange = ExchangeManager()
        exchange.connect()

        # 获取行情
        ticker = exchange.get_ticker("BTC/USDT:USDT")

        # 下单
        result = exchange.create_market_order("BTC/USDT:USDT", "buy", 0.1)
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

        self._exchange: Optional[ccxt.Exchange] = None
        self._config: Dict[str, Any] = {}
        self._market_info: Dict[str, Any] = {}
        self._connected = False
        self._logger = get_logger("exchange")
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._initialized = True

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected and self._exchange is not None

    @property
    def exchange(self) -> ccxt.Exchange:
        """获取交易所实例"""
        if not self._exchange:
            raise ConnectionError("交易所未连接")
        return self._exchange

    @log_execution_time(threshold_ms=5000)
    def connect(self) -> bool:
        """连接交易所

        Returns:
            是否连接成功
        """
        try:
            # 加载配置
            self._config = get_config("exchange")
            trading_config = get_config("trading")

            exchange_name = self._config.get("name", "okx")
            sandbox_mode = self._config.get("sandbox", False)

            self._logger.info(f"正在连接交易所: {exchange_name}")
            if sandbox_mode:
                self._logger.info("🎮 模拟盘模式已启用")

            # 构建交易所配置
            exchange_config = {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",  # 永续合约
                },
                "apiKey": self._config.get("api_key"),
                "secret": self._config.get("secret"),
            }

            # OKX 需要密码
            if exchange_name == "okx":
                exchange_config["password"] = self._config.get("password")

            # 模拟盘模式
            if sandbox_mode:
                exchange_config["sandbox"] = True
                if exchange_name == "okx":
                    exchange_config["options"]["sandboxMode"] = True

            # 创建交易所实例
            exchange_class = getattr(ccxt, exchange_name)
            self._exchange = exchange_class(exchange_config)

            # 验证连接
            self._verify_connection()

            # 加载市场信息
            self._load_markets()

            # 设置交易参数
            self._setup_trading_params()

            self._connected = True
            self._reconnect_attempts = 0

            self._logger.info("✅ 交易所连接成功")
            return True

        except Exception as e:
            self._logger.error(f"交易所连接失败: {e}")
            self._connected = False
            return False

    def _verify_connection(self):
        """验证 API 连接"""
        try:
            server_time = self._exchange.fetch_time()
            server_dt = datetime.fromtimestamp(server_time / 1000)
            self._logger.info(
                f"✅ API连接验证成功，服务器时间: {server_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            raise ConnectionError(f"API连接验证失败: {e}")

    def _load_markets(self):
        """加载市场信息"""
        try:
            markets = self._exchange.load_markets()
            symbol = get_config("trading.symbol")

            if symbol not in markets:
                raise ExchangeError(f"交易对 {symbol} 不存在")

            market = markets[symbol]
            self._market_info = {
                "contract_size": float(market.get("contractSize", 1)),
                "min_amount": market.get("limits", {}).get("amount", {}).get("min", 0.01),
                "price_precision": market.get("precision", {}).get("price", 8),
                "amount_precision": market.get("precision", {}).get("amount", 2),
                "symbol": symbol,
            }

            self._logger.info(
                f"✅ 市场信息已加载: {symbol}, "
                f"合约乘数: {self._market_info['contract_size']}, "
                f"最小交易量: {self._market_info['min_amount']}"
            )

        except Exception as e:
            raise ExchangeError(f"加载市场信息失败: {e}")

    def _setup_trading_params(self):
        """设置交易参数"""
        try:
            symbol = get_config("trading.symbol")
            leverage = get_config("trading.leverage")
            margin_mode = get_config("trading.margin_mode")

            # 设置单向持仓模式
            try:
                self._exchange.set_position_mode(False, symbol)
                self._logger.info("✅ 已设置单向持仓模式")
            except Exception as e:
                self._logger.warning(f"设置单向持仓模式失败 (可能已设置): {e}")

            # 设置仓位模式（OKX 特有）
            if hasattr(self._exchange, "private_post_account_set_margin_mode"):
                try:
                    self._exchange.private_post_account_set_margin_mode(
                        {"marginMode": margin_mode}
                    )
                    self._logger.info(f"✅ 已设置{margin_mode}模式")
                except Exception as e:
                    if "already" not in str(e).lower():
                        self._logger.warning(f"设置仓位模式失败: {e}")

            # 设置杠杆
            try:
                result = self._exchange.set_leverage(
                    leverage, symbol, {"mgnMode": margin_mode}
                )
                self._logger.info(f"✅ 已设置杠杆: {leverage}x")
            except Exception as e:
                self._logger.warning(f"设置杠杆失败: {e}")

            # 验证杠杆设置
            self._verify_leverage(leverage, margin_mode)

        except Exception as e:
            self._logger.error(f"设置交易参数失败: {e}")

    def _verify_leverage(self, expected_leverage: int, margin_mode: str):
        """验证杠杆设置"""
        try:
            symbol = get_config("trading.symbol")
            # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
            inst_id = symbol.replace("/", "-").replace(":USDT", "-USDT-SWAP")

            leverage_info = self._exchange.private_get_account_leverage_info(
                {"mgnMode": margin_mode, "instId": inst_id}
            )

            actual_leverage = int(leverage_info["data"][0]["lever"])
            if actual_leverage != expected_leverage:
                self._logger.warning(
                    f"杠杆设置不一致: 期望 {expected_leverage}x, 实际 {actual_leverage}x"
                )
            else:
                self._logger.info(f"✅ 杠杆验证成功: {actual_leverage}x")

        except Exception as e:
            self._logger.warning(f"验证杠杆失败: {e}")

    @retry_on_failure(max_retries=3, delay=1)
    def get_ticker(self, symbol: str = None) -> Ticker:
        """获取行情数据

        Args:
            symbol: 交易对，None 则使用配置中的

        Returns:
            Ticker 对象
        """
        symbol = symbol or get_config("trading.symbol")
        ticker = self.exchange.fetch_ticker(symbol)

        return Ticker(
            symbol=symbol,
            bid=safe_float(ticker.get("bid")),
            ask=safe_float(ticker.get("ask")),
            last=safe_float(ticker.get("last")),
            high=safe_float(ticker.get("high")),
            low=safe_float(ticker.get("low")),
            volume=safe_float(ticker.get("baseVolume")),
            timestamp=datetime.now(),
        )

    @retry_on_failure(max_retries=3, delay=1)
    def get_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """获取 K 线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 数量

        Returns:
            K 线数据列表
        """
        symbol = symbol or get_config("trading.symbol")
        timeframe = timeframe or get_config("trading.timeframe")

        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        return [
            {
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
            for candle in ohlcv
        ]

    def get_balance(self, currency: str = "USDT") -> Dict[str, float]:
        """获取账户余额

        Args:
            currency: 币种

        Returns:
            余额信息
        """
        balance = self.exchange.fetch_balance()

        if currency in balance:
            return {
                "total": safe_float(balance[currency].get("total")),
                "free": safe_float(balance[currency].get("free")),
                "used": safe_float(balance[currency].get("used")),
            }

        return {"total": 0, "free": 0, "used": 0}

    def get_position(self, symbol: str = None) -> Optional[Position]:
        """获取当前持仓

        Args:
            symbol: 交易对

        Returns:
            Position 对象或 None
        """
        symbol = symbol or get_config("trading.symbol")
        positions = self.exchange.fetch_positions([symbol])

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
                        liquidation_price=safe_float(pos.get("liquidationPrice")),
                        timestamp=datetime.now(),
                    )

        return None

    @retry_on_failure(max_retries=2, delay=0.5)
    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """创建市价订单

        Args:
            symbol: 交易对
            side: 方向 ('buy' 或 'sell')
            size: 数量
            reduce_only: 是否只减仓

        Returns:
            OrderResult 对象
        """
        margin_mode = get_config("trading.margin_mode")

        params = {
            "mgnMode": margin_mode,
        }
        if reduce_only:
            params["reduceOnly"] = True

        try:
            self._logger.trade(
                action="CREATE_ORDER",
                symbol=symbol,
                side=side,
                price=0,  # 市价单
                size=size,
                order_type="market",
                reduce_only=reduce_only,
            )

            order = self.exchange.create_market_order(symbol, side, size, params=params)

            result = OrderResult(
                success=True,
                order_id=order.get("id"),
                symbol=order.get("symbol"),
                side=order.get("side"),
                type=order.get("type"),
                size=safe_float(order.get("amount")),
                filled_size=safe_float(order.get("filled")),
                average_price=safe_float(order.get("average")),
                fee=safe_float(order.get("fee", {}).get("cost")),
                timestamp=datetime.now(),
            )

            self._logger.info(
                f"✅ 订单执行成功: {side} {size} {symbol} @ {smart_price_format(result.average_price or 0)}"
            )

            return result

        except ccxt.InsufficientFunds as e:
            self._logger.error(f"资金不足: {e}")
            return OrderResult(success=False, error=f"资金不足: {e}")

        except ccxt.InvalidOrder as e:
            self._logger.error(f"无效订单: {e}")
            return OrderResult(success=False, error=f"无效订单: {e}")

        except Exception as e:
            self._logger.error(f"订单执行失败: {e}")
            return OrderResult(success=False, error=str(e))

    def close_position(
        self,
        symbol: str = None,
        position: Position = None,
    ) -> OrderResult:
        """平仓

        Args:
            symbol: 交易对
            position: 持仓信息

        Returns:
            OrderResult 对象
        """
        if position is None:
            position = self.get_position(symbol)

        if not position:
            self._logger.warning("无持仓可平")
            return OrderResult(success=False, error="无持仓可平")

        # 平仓方向与持仓方向相反
        close_side = "sell" if position.is_long else "buy"

        self._logger.info(
            f"平仓: {position.side}仓 {position.size} 张 @ 入场价 {smart_price_format(position.entry_price)}"
        )

        return self.create_market_order(
            symbol or position.symbol,
            close_side,
            position.size,
            reduce_only=True,
        )

    @property
    def market_info(self) -> Dict[str, Any]:
        """获取市场信息"""
        return self._market_info.copy()

    @property
    def contract_size(self) -> float:
        """获取合约乘数"""
        return self._market_info.get("contract_size", 1.0)

    @property
    def min_amount(self) -> float:
        """获取最小交易量"""
        return self._market_info.get("min_amount", 0.01)

    def disconnect(self):
        """断开连接"""
        if self._exchange:
            self._exchange.close()
            self._exchange = None
            self._connected = False
            self._logger.info("交易所连接已断开")


# 全局交易所实例
_exchange_manager: Optional[ExchangeManager] = None


def get_exchange() -> ExchangeManager:
    """获取全局交易所实例"""
    global _exchange_manager
    if _exchange_manager is None:
        _exchange_manager = ExchangeManager()
    return _exchange_manager