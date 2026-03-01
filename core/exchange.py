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
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 在导入 ccxt 之前设置代理（解决 Windows 中国大陆网络问题）
# 从环境变量或配置文件读取代理配置
_proxy_from_env = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
if not _proxy_from_env:
    # 尝试从 .env 文件加载
    try:
        from dotenv import load_dotenv

        load_dotenv()
        _proxy_from_env = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    except Exception:
        pass

# 如果仍然没有代理，尝试从 config.yaml 读取
if not _proxy_from_env:
    try:
        import yaml

        _config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(_config_path):
            with open(_config_path, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f)
                _proxy_from_env = _cfg.get("exchange", {}).get("proxy")
                # 调试输出
                print(f"[EXCHANGE_MODULE] 从 config.yaml 读取代理: {_proxy_from_env}")
    except Exception as e:
        print(f"[EXCHANGE_MODULE] 读取 config.yaml 失败: {e}")

# 设置代理环境变量（必须在导入 ccxt 之前）
if _proxy_from_env:
    os.environ["HTTP_PROXY"] = _proxy_from_env
    os.environ["HTTPS_PROXY"] = _proxy_from_env
    print(f"[EXCHANGE_MODULE] 代理已设置: {_proxy_from_env}")
else:
    print("[EXCHANGE_MODULE] 警告: 未找到代理配置")

import ccxt

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.decorators import retry_on_failure, rate_limit, log_execution_time
from utils.helpers import safe_float, smart_price_format

# 导入统一的 Position dataclass（避免重复定义）
from core.position import Position as PositionDataclass


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
    volume: float  # 24h成交量 (基础货币)
    quote_volume: float = 0.0  # 24h成交量 (计价货币)
    percentage: float = 0.0  # 24h涨跌幅百分比
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def spread(self) -> float:
        """价差"""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """价差百分比"""
        return (self.spread / self.last) * 100 if self.last > 0 else 0


# 使用 core.position 模块中的 Position dataclass
# 为保持向后兼容，创建一个别名
Position = PositionDataclass


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
        self._position_mode: Optional[str] = (
            None  # 持仓模式: long_short_mode / net_mode
        )
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
                self._logger.info("[SANDBOX] 模拟盘模式已启用")

            # 构建交易所配置
            exchange_config = {
                "enableRateLimit": True,
                "timeout": 30000,  # 30秒超时
                "options": {
                    "defaultType": "swap",  # 永续合约
                    "loadMarkets": False,  # 延迟加载，使用原生 API 获取
                },
                "apiKey": self._config.get("api_key"),
                "secret": self._config.get("secret"),
            }

            # 代理配置（中国大陆访问 OKX 需要）
            proxy = self._config.get("proxy")
            if proxy:
                # 设置环境变量代理（最可靠的方式）
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
                # 同时设置 ccxt 的 proxies 配置
                exchange_config["proxies"] = {
                    "http": proxy,
                    "https": proxy,
                }
                self._logger.info(f"[PROXY] 已配置代理: {proxy}")

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

            self._logger.info("[OK] 交易所连接成功")
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
                f"[OK] API连接验证成功，服务器时间: {server_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            raise ConnectionError(f"API连接验证失败: {e}")

    def _load_markets(self):
        """加载市场信息"""
        try:
            symbol = get_config("trading.symbol")

            # 对于 OKX，直接使用原生 API 获取单个交易对的市场信息
            # 避免 ccxt 解析所有市场数据时的 NoneType 错误
            exchange_name = self._config.get("name", "okx")
            if exchange_name == "okx":
                markets = self._load_market_via_native_api()
            else:
                # 其他交易所使用 ccxt 默认方式
                try:
                    markets = self._exchange.load_markets()
                except Exception as e:
                    self._logger.warning(f"加载市场数据失败，尝试使用备选方法: {e}")
                    markets = self._load_market_via_native_api()

            if symbol not in markets:
                raise ExchangeError(f"交易对 {symbol} 不存在")

            market = markets[symbol]
            self._market_info = {
                "contract_size": float(market.get("contractSize", 1)),
                "min_amount": market.get("limits", {})
                .get("amount", {})
                .get("min", 0.01),
                "price_precision": market.get("precision", {}).get("price", 8),
                "amount_precision": market.get("precision", {}).get("amount", 2),
                "symbol": symbol,
            }

            self._logger.info(
                f"[OK] 市场信息已加载: {symbol}, "
                f"合约乘数: {self._market_info['contract_size']}, "
                f"最小交易量: {self._market_info['min_amount']}"
            )

        except Exception as e:
            raise ExchangeError(f"加载市场信息失败: {e}")

    def _load_market_via_native_api(self) -> Dict[str, Any]:
        """使用 OKX 原生 API 获取市场信息

        Returns:
            市场信息字典
        """
        symbol = get_config("trading.symbol")
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        try:
            # 获取交易对详情
            response = self._exchange.public_get_public_instruments(
                {"instType": "SWAP", "instId": inst_id}
            )

            if response.get("code") != "0":
                raise ExchangeError(f"获取市场信息失败: {response.get('msg')}")

            data = response.get("data", [])
            if not data:
                raise ExchangeError(f"未找到交易对 {inst_id}")

            market_data = data[0]

            # 构建符合 ccxt 格式的市场信息
            markets = {
                symbol: {
                    "id": inst_id,
                    "symbol": symbol,
                    "base": symbol.split("/")[0],
                    "quote": "USDT",
                    "settle": "USDT",
                    "type": "swap",
                    "contractSize": float(market_data.get("ctVal", 1)),
                    "contract": True,
                    "linear": True,
                    "limits": {
                        "amount": {
                            "min": float(market_data.get("minSz", 0.01)),
                            "max": float(market_data.get("maxSz", 10000)),
                        },
                        "price": {
                            "min": 0.01,
                            "max": 1000000,
                        },
                    },
                    "precision": {
                        "price": int(market_data.get("tickSz", "0.1").count("0") - 1)
                        if "." in market_data.get("tickSz", "0.1")
                        else 8,
                        "amount": int(market_data.get("lotSz", "0.01").count("0") - 1)
                        if "." in market_data.get("lotSz", "0.01")
                        else 2,
                    },
                }
            }

            # 手动设置交易所的市场数据，避免后续重复加载
            self._exchange.markets = markets
            self._exchange.markets_by_id = {inst_id: markets[symbol]}

            self._logger.info(f"[OK] 通过原生 API 加载市场信息: {symbol}")
            return markets

        except Exception as e:
            self._logger.error(f"原生 API 加载市场信息失败: {e}")
            # 返回默认市场信息
            markets = {
                symbol: {
                    "id": inst_id,
                    "symbol": symbol,
                    "contractSize": 1.0,
                    "limits": {"amount": {"min": 0.01}},
                    "precision": {"price": 8, "amount": 2},
                }
            }
            self._exchange.markets = markets
            self._exchange.markets_by_id = {inst_id: markets[symbol]}
            return markets

    def _setup_trading_params(self):
        """设置交易参数（优化版：减少不必要的 API 调用）"""
        try:
            symbol = get_config("trading.symbol")
            leverage = get_config("trading.leverage")
            margin_mode = get_config("trading.margin_mode")

            # 检测账户持仓模式
            self._detect_position_mode()

            # 仅设置杠杆，其他参数使用默认值或已有设置
            # 持仓模式和保证金模式通常在账户级别已设置，无需每次修改
            leverage_set = self._set_leverage_via_native_api(
                symbol, leverage, margin_mode
            )

            if leverage_set:
                self._logger.info(f"[OK] 已设置杠杆: {leverage}x")

        except Exception as e:
            self._logger.error(f"设置交易参数失败: {e}")

    def _detect_position_mode(self):
        """检测账户持仓模式

        OKX 有两种持仓模式:
        - long_short_mode: 双向持仓，需要 posSide 参数
        - net_mode: 单向持仓，不需要 posSide 参数
        """
        try:
            # 获取账户配置
            response = self._exchange.private_get_account_config()

            if response.get("code") == "0":
                data = response.get("data", [])
                if data:
                    self._position_mode = data[0].get("posMode", "net_mode")
                    self._logger.info(f"[OK] 账户持仓模式: {self._position_mode}")
                    return

            self._position_mode = "net_mode"  # 默认单向持仓
            self._logger.info(f"[OK] 使用默认持仓模式: {self._position_mode}")

        except Exception as e:
            self._position_mode = "net_mode"  # 默认单向持仓
            self._logger.warning(f"检测持仓模式失败，使用默认单向模式: {e}")

    def _set_leverage_via_native_api(
        self, symbol: str, leverage: int, margin_mode: str
    ) -> bool:
        """使用 OKX 原生 API 设置杠杆

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)
            leverage: 杠杆倍数
            margin_mode: 仓位模式 (cross/isolated)

        Returns:
            是否设置成功
        """
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        try:
            # OKX 设置杠杆 API
            # 文档: https://www.okx.com/docs-v5/zh/#trading-account-set-leverage
            params = {
                "instId": inst_id,
                "lever": str(leverage),
                "mgnMode": margin_mode,
            }

            # 逐仓模式需要指定 posSide
            if margin_mode == "isolated":
                # 单向持仓模式下，需要分别设置多头和空头的杠杆
                for pos_side in ["long", "short"]:
                    try:
                        params["posSide"] = pos_side
                        result = self._exchange.private_post_account_set_leverage(
                            params
                        )
                        if result.get("code") == "0":
                            self._logger.debug(f"设置 {pos_side} 杠杆成功: {leverage}x")
                    except Exception as e:
                        error_msg = str(e)
                        # 忽略已设置的错误
                        if "51001" in error_msg or "already" in error_msg.lower():
                            pass
                        elif "59000" in error_msg:
                            # 有持仓时设置杠杆需要特殊处理
                            self._logger.debug(f"有持仓，{pos_side} 杠杆设置跳过")
                        else:
                            self._logger.debug(f"设置 {pos_side} 杠杆: {e}")
                return True
            else:
                # 全仓模式
                result = self._exchange.private_post_account_set_leverage(params)
                if result.get("code") == "0":
                    return True
                else:
                    self._logger.warning(f"设置杠杆失败: {result.get('msg')}")
                    return False

        except Exception as e:
            error_msg = str(e)
            if "51001" in error_msg:
                # API Key 环境问题，静默忽略
                return True
            self._logger.warning(f"设置杠杆异常: {e}")
            return False

    def _verify_leverage(self, expected_leverage: int, margin_mode: str):
        """验证杠杆设置"""
        try:
            symbol = get_config("trading.symbol")
            # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
            inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

            leverage_info = self._exchange.private_get_account_leverage_info(
                {"mgnMode": margin_mode, "instId": inst_id}
            )

            if leverage_info.get("code") != "0":
                self._logger.debug(f"验证杠杆: {leverage_info.get('msg')}")
                return

            data = leverage_info.get("data", [])
            if data:
                actual_leverage = int(data[0].get("lever", 0))
                if actual_leverage != expected_leverage:
                    self._logger.warning(
                        f"杠杆设置不一致: 期望 {expected_leverage}x, 实际 {actual_leverage}x"
                    )
                else:
                    self._logger.info(f"[OK] 杠杆验证成功: {actual_leverage}x")

        except Exception as e:
            # 静默忽略验证错误，不影响交易
            self._logger.debug(f"验证杠杆跳过: {e}")

    @retry_on_failure(max_retries=3, delay=1)
    def get_ticker(self, symbol: str = None) -> Ticker:
        """获取行情数据

        Args:
            symbol: 交易对，None 则使用配置中的

        Returns:
            Ticker 对象
        """
        symbol = symbol or get_config("trading.symbol")

        # 直接使用 OKX 原生 API 获取行情，避免 ccxt 解析错误
        # ccxt 的 safe_market 函数在市场数据不完整时会抛出 KeyError
        try:
            ticker = self._get_ticker_via_native_api(symbol)
        except Exception as e:
            self._logger.warning(f"原生 API 获取行情失败，尝试 ccxt: {e}")
            try:
                ticker = self.exchange.fetch_ticker(symbol)
            except Exception as e2:
                raise ExchangeError(f"获取行情失败: {e2}")

        return Ticker(
            symbol=symbol,
            bid=safe_float(ticker.get("bid")),
            ask=safe_float(ticker.get("ask")),
            last=safe_float(ticker.get("last")),
            high=safe_float(ticker.get("high")),
            low=safe_float(ticker.get("low")),
            volume=safe_float(ticker.get("baseVolume")),
            quote_volume=safe_float(ticker.get("quoteVolume")),
            percentage=safe_float(ticker.get("percentage")),
            timestamp=datetime.now(),
        )

    def _get_ticker_via_native_api(self, symbol: str) -> Dict[str, Any]:
        """使用 OKX 原生 API 获取行情数据

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)

        Returns:
            行情数据字典
        """
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        # 调用 OKX 公共 API - 获取单个交易对行情
        response = self._exchange.public_get_market_ticker({"instId": inst_id})

        if response.get("code") != "0":
            raise ExchangeError(f"获取行情失败: {response.get('msg')}")

        data = response.get("data", [])
        if not data:
            raise ExchangeError(f"未找到交易对 {inst_id} 的行情数据")

        ticker_data = data[0]

        return {
            "bid": safe_float(ticker_data.get("bidPx")),
            "ask": safe_float(ticker_data.get("askPx")),
            "last": safe_float(ticker_data.get("last")),
            "high": safe_float(ticker_data.get("high24h")),
            "low": safe_float(ticker_data.get("low24h")),
            "baseVolume": safe_float(ticker_data.get("vol24h")),
            "quoteVolume": safe_float(ticker_data.get("volCcy24h")),
            "percentage": safe_float(ticker_data.get("change24h")),
        }

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

        # 使用原生 API 获取 K 线数据
        try:
            ohlcv = self._get_ohlcv_via_native_api(symbol, timeframe, limit)
        except Exception as e:
            self._logger.warning(f"原生 API 获取 K 线失败，尝试 ccxt: {e}")
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as e2:
                raise ExchangeError(f"获取 K 线失败: {e2}")

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

    def _get_ohlcv_via_native_api(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List]:
        """使用 OKX 原生 API 获取 K 线数据

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)
            timeframe: 时间周期 (如 1h, 4h, 1d)
            limit: 数量

        Returns:
            K 线数据列表 [[ts, open, high, low, close, volume], ...]
        """
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        # 转换时间周期格式: 1h -> 1H, 4h -> 4H, 1d -> 1D
        bar = timeframe.upper()

        # 调用 OKX 公共 API - 获取 K 线数据
        response = self._exchange.public_get_market_candles(
            {
                "instId": inst_id,
                "bar": bar,
                "limit": str(limit),
            }
        )

        if response.get("code") != "0":
            raise ExchangeError(f"获取 K 线失败: {response.get('msg')}")

        data = response.get("data", [])
        if not data:
            return []

        # OKX K 线数据格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        # ccxt 格式: [ts, o, h, l, c, vol]
        ohlcv = []
        for candle in data:
            # OKX 返回的时间戳是毫秒
            ts = int(candle[0])
            o = float(candle[1])
            h = float(candle[2])
            l = float(candle[3])
            c = float(candle[4])
            vol = float(candle[5])
            ohlcv.append([ts, o, h, l, c, vol])

        # OKX 返回的是倒序，需要反转
        ohlcv.reverse()

        return ohlcv

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
            symbol: 交易对，None 则获取所有 SWAP 持仓中的第一个

        Returns:
            Position 对象或 None（无持仓时返回 None）
        """
        symbol = symbol or get_config("trading.symbol")

        # 使用 OKX 原生 API 获取持仓
        position = self._get_position_via_native_api(symbol)
        return position

    def _get_position_via_native_api(self, symbol: str) -> Optional[Position]:
        """使用 OKX 原生 API 获取持仓

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)

        Returns:
            Position 对象或 None（无持仓时返回 None，不抛异常）
        """
        try:
            # 直接获取所有 SWAP 持仓（避免 instId 查询导致的认证问题）
            self._logger.debug("正在获取所有 SWAP 持仓...")

            # 添加重试机制
            max_retries = 3
            retry_delay = 1
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = self._exchange.private_get_account_positions(
                        {"instType": "SWAP"}
                    )
                    break  # 成功则跳出重试循环
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    # 如果是认证错误，不重试
                    if "50101" in error_msg:
                        self._logger.warning("API Key 环境不匹配，请检查配置")
                        return None
                    # 网络错误则重试
                    if attempt < max_retries - 1:
                        self._logger.debug(
                            f"获取持仓失败，重试 {attempt + 1}/{max_retries}: {e}"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        raise

            # 检查 API 响应
            if response is None:
                self._logger.warning("获取持仓响应为空")
                return None

            code = response.get("code")
            if code != "0":
                msg = response.get("msg", "Unknown error")
                self._logger.warning(f"获取持仓 API 返回错误: code={code}, msg={msg}")
                return None

            data = response.get("data", [])
            self._logger.debug(f"SWAP 持仓数据: {len(data)} 条")

            if not data:
                self._logger.debug("无任何持仓数据")
                return None

            # 遍历找到有持仓的数据
            for pos_data in data:
                pos_size = safe_float(pos_data.get("pos"))
                self._logger.debug(
                    f"检查持仓: instId={pos_data.get('instId')}, pos={pos_size}"
                )
                if pos_size == 0:
                    continue

                # 解析持仓信息
                pos_inst_id = pos_data.get("instId", "")
                # 转换 instId 为 symbol 格式: DOGE-USDT-SWAP -> DOGE/USDT:USDT
                # 先移除 -SWAP 后缀，然后将 -USDT- 替换为 /USDT:USDT
                pos_symbol = pos_inst_id.replace("-SWAP", "")
                # 处理 XXX-USDT-SWAP 格式 -> XXX/USDT:USDT
                if "-USDT" in pos_symbol:
                    pos_symbol = pos_symbol.replace("-USDT", "/USDT:USDT")
                # 处理其他格式如 XXX-USD-SWAP -> XXX/USD:USD
                elif "-USD" in pos_symbol and "-USDT" not in pos_symbol:
                    pos_symbol = pos_symbol.replace("-USD", "/USD:USD")

                # OKX 持仓方向: long/short/net
                pos_side = pos_data.get("posSide")
                if not pos_side or pos_side == "net":
                    pos_side = "long" if pos_size > 0 else "short"
                    pos_size = abs(pos_size)

                self._logger.info(
                    f"发现持仓: {pos_inst_id} -> {pos_symbol}, "
                    f"方向={pos_side}, 数量={pos_size}"
                )

                position = Position(
                    symbol=pos_symbol,
                    side=pos_side,
                    size=pos_size,
                    entry_price=safe_float(pos_data.get("avgPx")),
                    unrealized_pnl=safe_float(pos_data.get("upl")),
                    leverage=safe_float(
                        pos_data.get("lever", get_config("trading.leverage"))
                    ),
                    margin_mode=pos_data.get(
                        "mgnMode", get_config("trading.margin_mode")
                    ),
                    liquidation_price=safe_float(pos_data.get("liqPx")),
                    timestamp=datetime.now(),
                )

                # 如果找到的是配置的交易对，直接返回
                if pos_symbol == symbol:
                    return position

                # 否则返回第一个有持仓的（并记录日志）
                self._logger.info(f"返回其他交易对的持仓: {pos_symbol}")
                return position

            self._logger.debug("遍历完成，未找到有效持仓")
            return None

        except Exception as e:
            self._logger.warning(f"获取持仓异常: {type(e).__name__}: {e}")
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

        # 构建OKX原生API参数
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        # OKX 原生参数
        params = {
            "instId": inst_id,
            "tdMode": margin_mode == "isolated" and "isolated" or "cross",  # 交易模式
        }

        # 逐仓模式需要指定持仓方向
        if margin_mode == "isolated":
            # 根据订单方向确定持仓方向
            params["posSide"] = "long" if side == "buy" else "short"

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

            # 使用OKX原生API下单，避免ccxt市场类型解析问题
            order = self._create_order_via_native_api(symbol, side, size, params)

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
                f"[OK] 订单执行成功: {side} {size} {symbol} @ {smart_price_format(result.average_price or 0)}"
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

    def _create_order_via_native_api(
        self, symbol: str, side: str, size: float, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 OKX 原生 API 创建订单

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)
            side: 方向 ('buy' 或 'sell')
            size: 数量
            params: OKX 原生参数

        Returns:
            订单信息字典
        """
        # 转换交易对格式: BTC/USDT:USDT -> BTC-USDT-SWAP
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")

        # 构建下单请求参数
        order_params = {
            "instId": inst_id,
            "tdMode": params.get("tdMode", "cross"),  # 交易模式: cross/isolated
            "side": side,  # buy/sell
            "ordType": "market",  # 市价单
            "sz": str(size),  # 数量
        }

        # 根据持仓模式决定是否需要 posSide 参数
        # long_short_mode: 双向持仓，需要 posSide
        # net_mode: 单向持仓，不需要 posSide
        if self._position_mode == "long_short_mode":
            # 双向持仓模式：需要指定持仓方向
            order_params["posSide"] = params.get(
                "posSide", "long" if side == "buy" else "short"
            )
        # 单向持仓模式不需要 posSide 参数

        self._logger.debug(
            f"下单参数: instId={inst_id}, side={side}, sz={size}, "
            f"tdMode={order_params.get('tdMode')}, posMode={self._position_mode}"
        )

        # 调用 OKX 下单 API
        response = self._exchange.private_post_trade_order(order_params)

        if response.get("code") != "0":
            error_msg = response.get("msg", "Unknown error")
            # 检查是否是 posSide 错误，尝试切换模式重试
            error_data = response.get("data", [])
            if error_data:
                s_code = error_data[0].get("sCode", "")
                if s_code == "51000":
                    # posSide 错误，可能持仓模式判断有误，尝试不传 posSide
                    self._logger.warning("posSide 参数错误，尝试不使用 posSide 参数")
                    if "posSide" in order_params:
                        del order_params["posSide"]
                        self._position_mode = "net_mode"  # 更新模式
                        response = self._exchange.private_post_trade_order(order_params)
                        if response.get("code") == "0":
                            self._logger.info("[OK] 下单成功（使用单向持仓模式）")
                        else:
                            error_msg = response.get("msg", "Unknown error")
                            raise OrderError(f"下单失败: {error_msg}")
                    else:
                        raise OrderError(f"下单失败: {error_msg}")
                else:
                    raise OrderError(f"下单失败: {error_msg}")
            else:
                raise OrderError(f"下单失败: {error_msg}")

        data = response.get("data", [])
        if not data:
            raise OrderError("下单响应数据为空")

        order_data = data[0]

        # 返回符合 ccxt 格式的订单信息
        return {
            "id": order_data.get("ordId"),
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": size,
            "filled": size,  # 市价单默认全部成交
            "average": safe_float(order_data.get("avgPx")),
            "fee": {"cost": safe_float(order_data.get("fee"))},
        }

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

    def set_leverage(self, leverage: int, symbol: str = None) -> bool:
        """设置杠杆倍数

        Args:
            leverage: 杠杆倍数
            symbol: 交易对，None 则使用配置中的

        Returns:
            是否设置成功
        """
        symbol = symbol or get_config("trading.symbol")
        margin_mode = get_config("trading.margin_mode")
        return self._set_leverage_via_native_api(symbol, leverage, margin_mode)

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


# Alias for backward compatibility
get_exchange_manager = get_exchange
