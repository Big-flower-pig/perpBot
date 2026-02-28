"""
PerpBot 辅助函数模块

提供常用的辅助函数：
- 价格格式化
- 百分比格式化
- 时间戳格式化
- 盈亏计算
- 安全类型转换
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Union
import math


def format_price(
    price: float,
    decimals: int = None,
    currency: str = "$",
) -> str:
    """格式化价格

    Args:
        price: 价格
        decimals: 小数位数，None 则自动判断
        currency: 货币符号

    Returns:
        格式化后的价格字符串

    Example:
        format_price(1234.5678)  # "$1,234.57"
        format_price(0.123456)   # "$0.1235"
    """
    if price is None or math.isnan(price):
        return "N/A"

    # 自动判断小数位数
    if decimals is None:
        if price >= 1000:
            decimals = 2
        elif price >= 1:
            decimals = 4
        elif price >= 0.01:
            decimals = 6
        else:
            decimals = 8

    return f"{currency}{price:,.{decimals}f}"


def smart_price_format(price: float) -> str:
    """智能价格格式化

    根据价格大小自动选择合适的精度

    Args:
        price: 价格

    Returns:
        格式化后的价格字符串
    """
    if price is None or math.isnan(price):
        return "N/A"

    if price >= 10000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    elif price >= 0.01:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"


def format_percentage(
    value: float,
    decimals: int = 2,
    show_sign: bool = True,
) -> str:
    """格式化百分比

    Args:
        value: 数值
        decimals: 小数位数
        show_sign: 是否显示正负号

    Returns:
        格式化后的百分比字符串

    Example:
        format_percentage(0.1234)    # "+12.34%"
        format_percentage(-5.67)    # "-5.67%"
    """
    if value is None or math.isnan(value):
        return "N/A"

    if show_sign:
        return f"{value:+,.{decimals}f}%"
    return f"{value:,.{decimals}f}%"


def format_timestamp(
    timestamp: Union[int, float, datetime],
    fmt: str = "%Y-%m-%d %H:%M:%S",
    timezone_name: str = "Asia/Shanghai",
) -> str:
    """格式化时间戳

    Args:
        timestamp: 时间戳（毫秒/秒）或 datetime 对象
        fmt: 输出格式
        timezone_name: 时区名称

    Returns:
        格式化后的时间字符串
    """
    if timestamp is None:
        return "N/A"

    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        # 判断是毫秒还是秒
        if timestamp > 1e10:  # 毫秒
            timestamp = timestamp / 1000
        dt = datetime.fromtimestamp(timestamp)

    # 转换时区
    try:
        import pytz
        tz = pytz.timezone(timezone_name)
        dt = dt.astimezone(tz)
    except ImportError:
        pass

    return dt.strftime(fmt)


def format_duration(seconds: float) -> str:
    """格式化持续时间

    Args:
        seconds: 秒数

    Returns:
        格式化后的持续时间字符串

    Example:
        format_duration(3661)  # "1h 1m 1s"
    """
    if seconds < 0:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_number(
    value: float,
    decimals: int = 2,
    compact: bool = False,
) -> str:
    """格式化数字

    Args:
        value: 数值
        decimals: 小数位数
        compact: 是否使用紧凑格式（K、M、B）

    Returns:
        格式化后的数字字符串

    Example:
        format_number(1234567, compact=True)  # "1.23M"
    """
    if value is None or math.isnan(value):
        return "N/A"

    if not compact:
        return f"{value:,.{decimals}f}"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1e9:
        return f"{sign}{abs_value / 1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"{sign}{abs_value / 1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"{sign}{abs_value / 1e3:.{decimals}f}K"
    else:
        return f"{sign}{abs_value:.{decimals}f}"


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    side: str,
    leverage: float = 1.0,
    contract_size: float = 1.0,
) -> dict:
    """计算盈亏

    Args:
        entry_price: 入场价格
        exit_price: 出场价格
        size: 仓位大小（合约张数）
        side: 方向 ('long' 或 'short')
        leverage: 杠杆倍数
        contract_size: 合约乘数

    Returns:
        包含盈亏信息的字典

    Example:
        calculate_pnl(100, 110, 10, 'long')
    """
    # 价格变化百分比
    price_change_pct = (exit_price - entry_price) / entry_price * 100

    # 根据方向调整
    if side == "long":
        pnl_pct = price_change_pct
    else:  # short
        pnl_pct = -price_change_pct

    # 计算实际盈亏
    position_value = size * entry_price * contract_size
    pnl_usdt = position_value * pnl_pct / 100

    # 计算ROI（考虑杠杆）
    roi = pnl_pct * leverage

    return {
        "pnl_usdt": pnl_usdt,
        "pnl_pct": pnl_pct,
        "roi": roi,
        "price_change_pct": price_change_pct,
        "position_value": position_value,
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数

    Args:
        value: 任意值
        default: 默认值

    Returns:
        浮点数或默认值
    """
    if value is None:
        return default

    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数

    Args:
        value: 任意值
        default: 默认值

    Returns:
        整数或默认值
    """
    if value is None:
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """安全除法

    Args:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认值

    Returns:
        除法结果或默认值
    """
    if denominator == 0:
        return default
    return numerator / denominator


def round_to_tick(price: float, tick_size: float) -> float:
    """按最小变动单位取整

    Args:
        price: 价格
        tick_size: 最小变动单位

    Returns:
        取整后的价格
    """
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


def round_to_precision(value: float, precision: int) -> float:
    """按精度取整

    Args:
        value: 数值
        precision: 小数位数

    Returns:
        取整后的数值
    """
    return round(value, precision)


def calculate_position_size(
    capital: float,
    leverage: float,
    price: float,
    contract_size: float = 1.0,
    risk_ratio: float = 1.0,
) -> float:
    """计算仓位大小

    Args:
        capital: 本金
        leverage: 杠杆倍数
        price: 当前价格
        contract_size: 合约乘数
        risk_ratio: 风险比例（0-1）

    Returns:
        合约张数
    """
    # 公式：合约张数 = (本金 × 杠杆 × 风险比例) ÷ (价格 × 合约乘数)
    size = (capital * leverage * risk_ratio) / (price * contract_size)
    return round(size, 2)


def calculate_liquidation_price(
    entry_price: float,
    leverage: float,
    side: str,
    maintenance_margin_rate: float = 0.005,
) -> float:
    """计算强平价格

    Args:
        entry_price: 入场价格
        leverage: 杠杆倍数
        side: 方向 ('long' 或 'short')
        maintenance_margin_rate: 维持保证金率

    Returns:
        强平价格
    """
    if side == "long":
        # 多仓强平价 = 入场价 × (1 - 1/杠杆 + 维持保证金率)
        liquidation_price = entry_price * (1 - 1 / leverage + maintenance_margin_rate)
    else:
        # 空仓强平价 = 入场价 × (1 + 1/杠杆 - 维持保证金率)
        liquidation_price = entry_price * (1 + 1 / leverage - maintenance_margin_rate)

    return liquidation_price


def calculate_sharpe_ratio(
    returns: list,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """计算夏普比率

    Args:
        returns: 收益率列表
        risk_free_rate: 无风险利率
        periods_per_year: 年化周期数

    Returns:
        夏普比率
    """
    if not returns:
        return 0.0

    import statistics

    avg_return = statistics.mean(returns)
    std_return = statistics.stdev(returns) if len(returns) > 1 else 0

    if std_return == 0:
        return 0.0

    # 年化
    excess_return = (avg_return - risk_free_rate / periods_per_year) * periods_per_year
    annualized_std = std_return * (periods_per_year ** 0.5)

    return excess_return / annualized_std


def calculate_max_drawdown(values: list) -> dict:
    """计算最大回撤

    Args:
        values: 净值列表

    Returns:
        包含最大回撤信息的字典
    """
    if not values:
        return {"max_drawdown": 0, "peak_index": 0, "trough_index": 0}

    peak = values[0]
    max_dd = 0
    peak_idx = 0
    trough_idx = 0

    current_peak_idx = 0

    for i, value in enumerate(values):
        if value > peak:
            peak = value
            current_peak_idx = i

        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            peak_idx = current_peak_idx
            trough_idx = i

    return {
        "max_drawdown": max_dd * 100,
        "peak_index": peak_idx,
        "trough_index": trough_idx,
    }


def calculate_win_rate(trades: list) -> dict:
    """计算胜率

    Args:
        trades: 交易列表，每个交易包含 pnl 字段

    Returns:
        包含胜率信息的字典
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
        }

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]

    total_trades = len(trades)
    win_trades = len(wins)
    loss_trades = len(losses)

    total_win = sum(t.get("pnl", 0) for t in wins)
    total_loss = abs(sum(t.get("pnl", 0) for t in losses))

    avg_win = total_win / win_trades if win_trades > 0 else 0
    avg_loss = total_loss / loss_trades if loss_trades > 0 else 0

    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

    return {
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_trades / total_trades * 100 if total_trades > 0 else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def get_time_until_next_interval(
    interval_minutes: int = 15,
    current_time: datetime = None,
) -> dict:
    """计算到下一个时间间隔的等待时间

    Args:
        interval_minutes: 间隔分钟数
        current_time: 当前时间，None 则使用当前系统时间

    Returns:
        包含等待时间信息的字典
    """
    if current_time is None:
        current_time = datetime.now()

    current_minute = current_time.minute
    next_interval = ((current_minute // interval_minutes) + 1) * interval_minutes

    if next_interval >= 60:
        next_hour = current_time.hour + 1
        next_interval = 0
        if next_hour >= 24:
            next_hour = 0
    else:
        next_hour = current_time.hour

    next_time = current_time.replace(
        hour=next_hour if next_interval == 0 and current_minute > 0 else current_time.hour,
        minute=next_interval if next_interval < 60 else 0,
        second=0,
        microsecond=0,
    )

    if next_interval == 0 and current_minute > 0:
        next_time = next_time.replace(hour=current_time.hour + 1)

    wait_seconds = (next_time - current_time).total_seconds()

    return {
        "next_time": next_time,
        "wait_seconds": wait_seconds,
        "wait_minutes": wait_seconds / 60,
        "next_hour": next_time.hour,
        "next_minute": next_time.minute,
    }