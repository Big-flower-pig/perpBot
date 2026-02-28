"""
PerpBot 验证器模块

提供参数验证功能：
- 交易对验证
- 杠杆验证
- 方向验证
- 配置验证
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


class ValidationError(Exception):
    """验证错误异常"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class ValidationResult:
    """验证结果"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str):
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)


def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """验证交易对格式

    Args:
        symbol: 交易对字符串

    Returns:
        (是否有效, 错误消息)

    Example:
        validate_symbol("BTC/USDT:USDT")  # (True, "")
        validate_symbol("invalid")        # (False, "无效的交易对格式")
    """
    if not symbol:
        return False, "交易对不能为空"

    # OKX 永续合约格式: BTC/USDT:USDT
    pattern = r"^[A-Z]+/USDT:USDT$"
    if not re.match(pattern, symbol):
        return False, f"无效的交易对格式: {symbol}，应为 XXX/USDT:USDT"

    return True, ""


def validate_leverage(leverage: Union[int, float]) -> Tuple[bool, str]:
    """验证杠杆倍数

    Args:
        leverage: 杠杆倍数

    Returns:
        (是否有效, 错误消息)
    """
    if leverage is None:
        return False, "杠杆倍数不能为空"

    try:
        leverage = float(leverage)
    except (ValueError, TypeError):
        return False, "杠杆倍数必须是数字"

    if leverage < 1:
        return False, "杠杆倍数不能小于1"

    if leverage > 125:
        return False, "杠杆倍数不能超过125"

    return True, ""


def validate_side(side: str) -> Tuple[bool, str]:
    """验证交易方向

    Args:
        side: 方向字符串

    Returns:
        (是否有效, 错误消息)
    """
    valid_sides = ["long", "short", "buy", "sell"]

    if not side:
        return False, "方向不能为空"

    if side.lower() not in valid_sides:
        return False, f"无效的方向: {side}，应为 {valid_sides} 之一"

    return True, ""


def validate_signal(signal: str) -> Tuple[bool, str]:
    """验证交易信号

    Args:
        signal: 信号字符串

    Returns:
        (是否有效, 错误消息)
    """
    valid_signals = ["BUY", "SELL", "HOLD", "CLOSE"]

    if not signal:
        return False, "信号不能为空"

    if signal.upper() not in valid_signals:
        return False, f"无效的信号: {signal}，应为 {valid_signals} 之一"

    return True, ""


def validate_confidence(confidence: str) -> Tuple[bool, str]:
    """验证信心程度

    Args:
        confidence: 信心程度字符串

    Returns:
        (是否有效, 错误消息)
    """
    valid_levels = ["HIGH", "MEDIUM", "LOW"]

    if not confidence:
        return False, "信心程度不能为空"

    if confidence.upper() not in valid_levels:
        return False, f"无效的信心程度: {confidence}，应为 {valid_levels} 之一"

    return True, ""


def validate_margin_mode(mode: str) -> Tuple[bool, str]:
    """验证仓位模式

    Args:
        mode: 模式字符串

    Returns:
        (是否有效, 错误消息)
    """
    valid_modes = ["cross", "isolated"]

    if not mode:
        return False, "仓位模式不能为空"

    if mode.lower() not in valid_modes:
        return False, f"无效的仓位模式: {mode}，应为 {valid_modes} 之一"

    return True, ""


def validate_timeframe(timeframe: str) -> Tuple[bool, str]:
    """验证时间周期

    Args:
        timeframe: 时间周期字符串

    Returns:
        (是否有效, 错误消息)
    """
    valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    if not timeframe:
        return False, "时间周期不能为空"

    if timeframe.lower() not in valid_timeframes:
        return False, f"无效的时间周期: {timeframe}，应为 {valid_timeframes} 之一"

    return True, ""


def validate_positive_number(
    value: Union[int, float],
    field_name: str,
    min_value: float = None,
    max_value: float = None,
) -> Tuple[bool, str]:
    """验证正数

    Args:
        value: 数值
        field_name: 字段名称
        min_value: 最小值
        max_value: 最大值

    Returns:
        (是否有效, 错误消息)
    """
    if value is None:
        return False, f"{field_name}不能为空"

    try:
        value = float(value)
    except (ValueError, TypeError):
        return False, f"{field_name}必须是数字"

    if min_value is not None and value < min_value:
        return False, f"{field_name}不能小于{min_value}"

    if max_value is not None and value > max_value:
        return False, f"{field_name}不能大于{max_value}"

    return True, ""


def validate_price(
    price: float,
    field_name: str = "价格",
    allow_zero: bool = False,
) -> Tuple[bool, str]:
    """验证价格

    Args:
        price: 价格
        field_name: 字段名称
        allow_zero: 是否允许零

    Returns:
        (是否有效, 错误消息)
    """
    if price is None:
        return False, f"{field_name}不能为空"

    try:
        price = float(price)
    except (ValueError, TypeError):
        return False, f"{field_name}必须是数字"

    if allow_zero:
        if price < 0:
            return False, f"{field_name}不能为负数"
    else:
        if price <= 0:
            return False, f"{field_name}必须大于0"

    return True, ""


def validate_size(
    size: float,
    min_size: float = 0.01,
    field_name: str = "仓位大小",
) -> Tuple[bool, str]:
    """验证仓位大小

    Args:
        size: 仓位大小
        min_size: 最小值
        field_name: 字段名称

    Returns:
        (是否有效, 错误消息)
    """
    if size is None:
        return False, f"{field_name}不能为空"

    try:
        size = float(size)
    except (ValueError, TypeError):
        return False, f"{field_name}必须是数字"

    if size < min_size:
        return False, f"{field_name}不能小于{min_size}"

    return True, ""


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """验证完整配置

    Args:
        config: 配置字典

    Returns:
        验证结果
    """
    result = ValidationResult()

    # 验证交易所配置
    if "exchange" in config:
        exchange = config["exchange"]
        if "name" not in exchange:
            result.add_error("exchange.name 缺失")
        if "sandbox" not in exchange:
            result.add_warning("exchange.sandbox 未设置，默认为 False")

    # 验证交易配置
    if "trading" in config:
        trading = config["trading"]
        valid, msg = validate_symbol(trading.get("symbol", ""))
        if not valid:
            result.add_error(f"trading.symbol: {msg}")

        valid, msg = validate_leverage(trading.get("leverage", 0))
        if not valid:
            result.add_error(f"trading.leverage: {msg}")

        valid, msg = validate_margin_mode(trading.get("margin_mode", ""))
        if not valid:
            result.add_error(f"trading.margin_mode: {msg}")

        valid, msg = validate_timeframe(trading.get("timeframe", ""))
        if not valid:
            result.add_error(f"trading.timeframe: {msg}")

    # 验证仓位配置
    if "position" in config:
        position = config["position"]
        valid_modes = ["fixed", "compound"]
        if position.get("mode", "").lower() not in valid_modes:
            result.add_error(f"position.mode 无效")

        valid, msg = validate_positive_number(
            position.get("initial_capital"), "position.initial_capital", min_value=1
        )
        if not valid:
            result.add_error(msg)

    # 验证风险配置
    if "risk" in config:
        risk = config["risk"]
        valid, msg = validate_positive_number(
            risk.get("stop_loss_percent"), "risk.stop_loss_percent", min_value=0.1, max_value=50
        )
        if not valid:
            result.add_error(msg)

        valid, msg = validate_positive_number(
            risk.get("take_profit_percent"), "risk.take_profit_percent", min_value=0.1, max_value=100
        )
        if not valid:
            result.add_error(msg)

    # 验证监控配置
    if "monitoring" in config:
        monitoring = config["monitoring"]
        valid, msg = validate_positive_number(
            monitoring.get("check_interval"), "monitoring.check_interval", min_value=1
        )
        if not valid:
            result.add_error(msg)

    # 验证通知配置
    if "notification" in config:
        notification = config["notification"]
        if notification.get("telegram", {}).get("enabled"):
            telegram = notification["telegram"]
            if not telegram.get("bot_token"):
                result.add_error("notification.telegram.bot_token 缺失")
            if not telegram.get("chat_id"):
                result.add_error("notification.telegram.chat_id 缺失")

    return result


def validate_order_params(
    symbol: str,
    side: str,
    size: float,
    price: float = None,
    leverage: int = None,
) -> ValidationResult:
    """验证订单参数

    Args:
        symbol: 交易对
        side: 方向
        size: 大小
        price: 价格（可选）
        leverage: 杠杆（可选）

    Returns:
        验证结果
    """
    result = ValidationResult()

    valid, msg = validate_symbol(symbol)
    if not valid:
        result.add_error(f"symbol: {msg}")

    valid, msg = validate_side(side)
    if not valid:
        result.add_error(f"side: {msg}")

    valid, msg = validate_size(size)
    if not valid:
        result.add_error(f"size: {msg}")

    if price is not None:
        valid, msg = validate_price(price)
        if not valid:
            result.add_error(f"price: {msg}")

    if leverage is not None:
        valid, msg = validate_leverage(leverage)
        if not valid:
            result.add_error(f"leverage: {msg}")

    return result


def validate_signal_data(signal_data: Dict[str, Any]) -> ValidationResult:
    """验证信号数据

    Args:
        signal_data: 信号数据字典

    Returns:
        验证结果
    """
    result = ValidationResult()

    # 必需字段
    required_fields = ["signal", "reason", "stop_loss", "take_profit", "confidence"]
    for field in required_fields:
        if field not in signal_data:
            result.add_error(f"缺少必需字段: {field}")

    # 验证信号类型
    if "signal" in signal_data:
        valid, msg = validate_signal(signal_data["signal"])
        if not valid:
            result.add_error(f"signal: {msg}")

    # 验证信心程度
    if "confidence" in signal_data:
        valid, msg = validate_confidence(signal_data["confidence"])
        if not valid:
            result.add_error(f"confidence: {msg}")

    # 验证止损止盈
    if "stop_loss" in signal_data and signal_data["stop_loss"] is not None:
        valid, msg = validate_price(signal_data["stop_loss"], "stop_loss")
        if not valid:
            result.add_error(msg)

    if "take_profit" in signal_data and signal_data["take_profit"] is not None:
        valid, msg = validate_price(signal_data["take_profit"], "take_profit")
        if not valid:
            result.add_error(msg)

    return result


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """清理字符串

    Args:
        value: 字符串值
        max_length: 最大长度

    Returns:
        清理后的字符串
    """
    if not isinstance(value, str):
        return ""

    # 移除控制字符
    value = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")

    # 限制长度
    if len(value) > max_length:
        value = value[:max_length]

    return value.strip()


def validate_and_sanitize(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict, List[str]]:
    """验证并清理数据

    Args:
        data: 数据字典
        schema: 模式定义

    Returns:
        (清理后的数据, 错误列表)
    """
    errors = []
    result = {}

    for field, spec in schema.items():
        value = data.get(field)

        # 检查必需字段
        if spec.get("required", False) and value is None:
            errors.append(f"缺少必需字段: {field}")
            continue

        # 类型转换
        if value is not None:
            field_type = spec.get("type", str)
            try:
                if field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif field_type == str:
                    value = sanitize_string(str(value), spec.get("max_length", 1000))
                elif field_type == bool:
                    value = bool(value)
            except (ValueError, TypeError) as e:
                errors.append(f"字段 {field} 类型转换失败: {e}")
                continue

        # 范围检查
        if value is not None:
            if "min" in spec and value < spec["min"]:
                errors.append(f"字段 {field} 值 {value} 小于最小值 {spec['min']}")
            if "max" in spec and value > spec["max"]:
                errors.append(f"字段 {field} 值 {value} 大于最大值 {spec['max']}")

        # 自定义验证
        if value is not None and "validator" in spec:
            valid, msg = spec["validator"](value)
            if not valid:
                errors.append(f"字段 {field}: {msg}")

        # 默认值
        if value is None and "default" in spec:
            value = spec["default"]

        result[field] = value

    return result, errors