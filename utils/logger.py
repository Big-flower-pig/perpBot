"""
PerpBot 日志系统

提供完善的日志功能，支持：
- 多级别日志（DEBUG、INFO、WARNING、ERROR、CRITICAL）
- 日志文件轮转
- 彩色终端输出
- 结构化日志（JSON）
- 敏感信息脱敏
"""

import os
import sys
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading


# 颜色代码
class Colors:
    """终端颜色代码"""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 背景色
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


# 日志级别颜色映射
LEVEL_COLORS = {
    logging.DEBUG: Colors.CYAN,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
}

# 日志级别图标
LEVEL_ICONS = {
    logging.DEBUG: "🔍",
    logging.INFO: "✅",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🚨",
}


class SensitiveDataFilter:
    """敏感数据过滤器"""

    # 敏感字段列表
    SENSITIVE_KEYS = [
        "api_key",
        "apikey",
        "api_key_id",
        "secret",
        "password",
        "token",
        "private_key",
        "credential",
    ]

    # 替换字符串
    MASK = "****MASKED****"

    @classmethod
    def filter_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """过滤字典中的敏感信息"""
        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            if key.lower() in cls.SENSITIVE_KEYS:
                filtered[key] = cls.MASK
            elif isinstance(value, dict):
                filtered[key] = cls.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    cls.filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        colorize: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 过滤敏感信息
        if hasattr(record, "data") and isinstance(record.data, dict):
            record.data = SensitiveDataFilter.filter_dict(record.data)

        # 应用颜色
        if self.colorize and sys.stdout.isatty():
            color = LEVEL_COLORS.get(record.levelno, Colors.RESET)
            icon = LEVEL_ICONS.get(record.levelno, "")
            record.levelname = f"{color}{icon} {record.levelname}{Colors.RESET}"

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化为 JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外数据
        if hasattr(record, "data"):
            log_data["data"] = SensitiveDataFilter.filter_dict(record.data)

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class TradingLogger(logging.Logger):
    """交易日志器

    扩展标准 Logger，添加交易专用方法
    """

    def trade(
        self,
        action: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        **kwargs,
    ):
        """记录交易日志"""
        data = {
            "action": action,
            "symbol": symbol,
            "side": side,
            "price": price,
            "size": size,
            **kwargs,
        }
        self.info(f"[TRADE] {action} {side} {symbol} @ {price}", extra={"data": data})

    def signal(
        self,
        signal: str,
        symbol: str,
        confidence: str,
        reason: str,
        **kwargs,
    ):
        """记录信号日志"""
        data = {
            "signal": signal,
            "symbol": symbol,
            "confidence": confidence,
            "reason": reason,
            **kwargs,
        }
        self.info(f"[SIGNAL] {signal} {symbol} ({confidence})", extra={"data": data})

    def risk(
        self,
        event: str,
        level: str,
        details: Dict[str, Any],
    ):
        """记录风险事件"""
        data = {"event": event, "level": level, "details": details}
        if level == "CRITICAL":
            self.critical(f"[RISK] {event}", extra={"data": data})
        elif level == "HIGH":
            self.error(f"[RISK] {event}", extra={"data": data})
        elif level == "MEDIUM":
            self.warning(f"[RISK] {event}", extra={"data": data})
        else:
            self.info(f"[RISK] {event}", extra={"data": data})

    def performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs,
    ):
        """记录性能日志"""
        data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **kwargs,
        }
        status = "✓" if success else "✗"
        self.debug(
            f"[PERF] {operation} {status} ({duration_ms:.2f}ms)",
            extra={"data": data},
        )

    def ai_decision(
        self,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        decision: str,
    ):
        """记录 AI 决策日志"""
        data = {
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": duration_ms,
            "decision": decision,
        }
        self.info(
            f"[AI] {provider} decision: {decision} ({duration_ms:.0f}ms)",
            extra={"data": data},
        )


class TradingLoggerAdapter(logging.LoggerAdapter):
    """日志适配器，添加上下文信息"""

    def process(self, msg, kwargs):
        """处理日志消息，添加额外信息"""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)
        return msg, kwargs


class LogManager:
    """日志管理器

    单例模式，统一管理所有日志器
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._loggers: Dict[str, logging.Logger] = {}
        self._config: Dict[str, Any] = {}
        self._initialized = True

    def setup(
        self,
        level: str = "INFO",
        console: bool = True,
        file: bool = True,
        file_path: str = "logs/trading.log",
        max_size: str = "10MB",
        backup_count: int = 5,
        json_format: bool = False,
        colorize: bool = True,
    ):
        """设置日志配置"""
        # 解析日志级别
        log_level = getattr(logging, level.upper(), logging.INFO)

        # 解析文件大小
        size_multipliers = {
            "GB": 1024 * 1024 * 1024,
            "MB": 1024 * 1024,
            "KB": 1024,
            "B": 1,
        }
        max_bytes = 10 * 1024 * 1024  # 默认 10MB
        if isinstance(max_size, str):
            for suffix, multiplier in size_multipliers.items():
                if max_size.upper().endswith(suffix):
                    max_bytes = int(float(max_size[: -len(suffix)]) * multiplier)
                    break

        self._config = {
            "level": log_level,
            "console": console,
            "file": file,
            "file_path": file_path,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
            "json_format": json_format,
            "colorize": colorize,
        }

        # 设置根日志器
        self._setup_root_logger()

    def _setup_root_logger(self):
        """设置根日志器"""
        # 使用自定义日志器类
        logging.setLoggerClass(TradingLogger)
        root_logger = logging.getLogger("perpbot")
        root_logger.setLevel(self._config["level"])

        # 清除现有处理器
        root_logger.handlers.clear()

        # 控制台处理器
        if self._config["console"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._config["level"])
            if self._config["json_format"]:
                console_handler.setFormatter(JsonFormatter())
            else:
                console_handler.setFormatter(
                    ColoredFormatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        colorize=self._config["colorize"],
                    )
                )
            root_logger.addHandler(console_handler)

        # 文件处理器
        if self._config["file"]:
            # 确保目录存在
            log_path = Path(self._config["file_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                self._config["file_path"],
                maxBytes=self._config["max_bytes"],
                backupCount=self._config["backup_count"],
                encoding="utf-8",
            )
            file_handler.setLevel(self._config["level"])
            if self._config["json_format"]:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
            root_logger.addHandler(file_handler)

        # 防止日志传播到根日志器
        root_logger.propagate = False

    def get_logger(self, name: str) -> TradingLogger:
        """获取日志器"""
        full_name = f"perpbot.{name}"
        if full_name not in self._loggers:
            self._loggers[full_name] = logging.getLogger(full_name)
        return self._loggers[full_name]

    def add_handler(self, handler: logging.Handler, name: Optional[str] = None):
        """添加自定义处理器"""
        root_logger = logging.getLogger("perpbot")
        handler.name = name or f"custom_{id(handler)}"
        root_logger.addHandler(handler)


# 全局日志管理器实例
_log_manager = LogManager()


def setup_logger(
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    file_path: str = "logs/trading.log",
    max_size: str = "10MB",
    backup_count: int = 5,
    json_format: bool = False,
    colorize: bool = True,
):
    """设置日志系统

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: 是否输出到控制台
        file: 是否输出到文件
        file_path: 日志文件路径
        max_size: 单文件最大大小
        backup_count: 保留文件数量
        json_format: 是否使用 JSON 格式
        colorize: 是否彩色输出
    """
    _log_manager.setup(
        level=level,
        console=console,
        file=file,
        file_path=file_path,
        max_size=max_size,
        backup_count=backup_count,
        json_format=json_format,
        colorize=colorize,
    )


# Alias for backward compatibility
setup_logging = setup_logger


def get_logger(name: str) -> TradingLogger:
    """获取日志器

    Args:
        name: 模块名称

    Returns:
        TradingLogger 实例
    """
    return _log_manager.get_logger(name)


# 便捷函数
def debug(msg: str, **kwargs):
    """记录 DEBUG 级别日志"""
    get_logger("root").debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """记录 INFO 级别日志"""
    get_logger("root").info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """记录 WARNING 级别日志"""
    get_logger("root").warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """记录 ERROR 级别日志"""
    get_logger("root").error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """记录 CRITICAL 级别日志"""
    get_logger("root").critical(msg, **kwargs)
