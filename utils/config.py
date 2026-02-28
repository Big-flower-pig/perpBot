"""
PerpBot 配置管理模块

提供完善的配置管理功能：
- YAML 配置加载
- 配置验证（schema）
- 敏感信息加密
- 热重载支持
- 环境变量覆盖
"""

import os
import re
import json
import yaml
import threading
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import base64

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from dotenv import load_dotenv

# 类型变量
T = TypeVar("T")


@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class ConfigSchema:
    """配置模式定义"""

    # 必需的顶级配置项
    REQUIRED_KEYS = [
        "exchange",
        "trading",
        "ai",
        "position",
        "risk",
        "monitoring",
    ]

    # 配置项类型定义
    SCHEMA = {
        "exchange": {
            "type": dict,
            "required_keys": ["name", "sandbox", "api_key", "secret"],
        },
        "trading": {
            "type": dict,
            "required_keys": ["symbol", "leverage", "timeframe", "margin_mode"],
        },
        "ai": {
            "type": dict,
            "required_keys": ["provider", "api_key", "model"],
        },
        "position": {
            "type": dict,
            "required_keys": ["mode", "initial_capital"],
        },
        "risk": {
            "type": dict,
            "required_keys": ["stop_loss_percent", "take_profit_percent"],
        },
        "monitoring": {
            "type": dict,
            "required_keys": ["enabled", "check_interval"],
        },
    }

    # 数值范围限制
    RANGES = {
        "trading.leverage": (1, 125),
        "position.initial_capital": (1, 1000000),
        "risk.stop_loss_percent": (0.1, 50),
        "risk.take_profit_percent": (0.1, 100),
        "monitoring.check_interval": (1, 3600),
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> ConfigValidationResult:
        """验证配置"""
        result = ConfigValidationResult(valid=True)

        # 检查必需的顶级配置项
        for key in cls.REQUIRED_KEYS:
            if key not in config:
                result.errors.append(f"缺少必需配置项: {key}")
                result.valid = False

        # 检查各配置项的必需子项
        for key, spec in cls.SCHEMA.items():
            if key not in config:
                continue

            value = config[key]

            # 类型检查
            if not isinstance(value, spec["type"]):
                result.errors.append(
                    f"配置项 {key} 类型错误，期望 {spec['type'].__name__}"
                )
                result.valid = False
                continue

            # 必需子项检查
            if "required_keys" in spec:
                for sub_key in spec["required_keys"]:
                    if sub_key not in value:
                        # 检查是否是敏感信息（可能从环境变量读取）
                        if sub_key in ["api_key", "secret", "password"]:
                            continue
                        result.errors.append(f"配置项 {key}.{sub_key} 缺失")
                        result.valid = False

        # 数值范围检查
        for path, (min_val, max_val) in cls.RANGES.items():
            keys = path.split(".")
            value = config
            try:
                for key in keys:
                    value = value[key]
                if not (min_val <= value <= max_val):
                    result.warnings.append(
                        f"配置项 {path} 值 {value} 超出建议范围 [{min_val}, {max_val}]"
                    )
            except (KeyError, TypeError):
                pass

        return result


class EnvVarResolver:
    """环境变量解析器

    解析配置中的环境变量引用，格式：${ENV_VAR} 或 ${ENV_VAR:default}
    """

    PATTERN = re.compile(r"\$\{([^}]+)\}")

    @classmethod
    def resolve(cls, value: Any) -> Any:
        """递归解析环境变量"""
        if isinstance(value, str):
            return cls._resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item) for item in value]
        return value

    @classmethod
    def _resolve_string(cls, value: str) -> str:
        """解析字符串中的环境变量"""
        def replace(match):
            expr = match.group(1)
            # 检查是否有默认值
            if ":" in expr:
                var_name, default = expr.split(":", 1)
                return os.getenv(var_name, default)
            else:
                var_name = expr
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(f"环境变量 {var_name} 未设置")
                return env_value

        return cls.PATTERN.sub(replace, value)


class SecretManager:
    """敏感信息管理器"""

    def __init__(self, key: Optional[str] = None):
        """初始化

        Args:
            key: 加密密钥，如果不提供则自动生成
        """
        self._enabled = CRYPTO_AVAILABLE
        if self._enabled:
            if key:
                # 确保密钥是有效的 Fernet 密钥
                self._key = key.encode() if isinstance(key, str) else key
            else:
                self._key = Fernet.generate_key()
            self._cipher = Fernet(self._key)

    def encrypt(self, value: str) -> str:
        """加密值"""
        if not self._enabled:
            return value
        return self._cipher.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        """解密值"""
        if not self._enabled:
            return value
        return self._cipher.decrypt(value.encode()).decode()

    @staticmethod
    def hash_value(value: str) -> str:
        """生成哈希值（用于验证）"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]


class ConfigWatcher:
    """配置文件监视器

    监视配置文件变化并触发回调
    """

    def __init__(self, file_path: str, callback: callable):
        self.file_path = Path(file_path)
        self.callback = callback
        self._last_mtime = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, interval: float = 5.0):
        """开始监视"""
        if self._running:
            return

        self._last_mtime = self._get_mtime()
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, args=(interval,))
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """停止监视"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _get_mtime(self) -> float:
        """获取文件修改时间"""
        try:
            return self.file_path.stat().st_mtime
        except FileNotFoundError:
            return 0

    def _watch_loop(self, interval: float):
        """监视循环"""
        while self._running:
            current_mtime = self._get_mtime()
            if current_mtime > self._last_mtime:
                self._last_mtime = current_mtime
                try:
                    self.callback()
                except Exception as e:
                    # 避免回调异常影响监视
                    pass
            threading.Event().wait(interval)


class ConfigManager:
    """配置管理器

    单例模式，统一管理所有配置
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

        self._config: Dict[str, Any] = {}
        self._config_file: Optional[str] = None
        self._watcher: Optional[ConfigWatcher] = None
        self._secret_manager: Optional[SecretManager] = None
        self._change_callbacks: list = []
        self._initialized = True

    def load(
        self,
        config_file: str = "config.yaml",
        validate: bool = True,
        resolve_env: bool = True,
        watch: bool = False,
    ) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_file: 配置文件路径
            validate: 是否验证配置
            resolve_env: 是否解析环境变量
            watch: 是否监视文件变化

        Returns:
            配置字典
        """
        # 加载 .env 文件
        load_dotenv()

        # 检查文件存在
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        # 根据文件类型加载
        suffix = config_path.suffix.lower()
        with open(config_path, "r", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                self._config = yaml.safe_load(f)
            elif suffix == ".json":
                self._config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")

        # 解析环境变量
        if resolve_env:
            self._config = EnvVarResolver.resolve(self._config)

        # 验证配置
        if validate:
            result = ConfigSchema.validate(self._config)
            if not result.valid:
                raise ValueError(f"配置验证失败:\n" + "\n".join(result.errors))
            for warning in result.warnings:
                print(f"⚠️ 配置警告: {warning}")

        self._config_file = str(config_path)

        # 启动文件监视
        if watch:
            self._start_watcher()

        return self._config

    def _start_watcher(self):
        """启动配置文件监视"""
        if self._watcher:
            self._watcher.stop()

        self._watcher = ConfigWatcher(
            self._config_file,
            self._on_config_change,
        )
        self._watcher.start()

    def _on_config_change(self):
        """配置文件变化回调"""
        try:
            self.load(
                self._config_file,
                validate=True,
                resolve_env=True,
                watch=False,
            )
            # 触发回调
            for callback in self._change_callbacks:
                try:
                    callback(self._config)
                except Exception:
                    pass
        except Exception:
            pass

    def on_change(self, callback: callable):
        """注册配置变化回调"""
        self._change_callbacks.append(callback)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        支持点号分隔的路径，如: trading.symbol

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any, save: bool = False):
        """设置配置值

        Args:
            key: 配置键
            value: 配置值
            save: 是否保存到文件
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

        if save and self._config_file:
            self._save()

    def _save(self):
        """保存配置到文件"""
        if not self._config_file:
            return

        with open(self._config_file, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节"""
        return self._config.get(section, {})

    @property
    def trading(self) -> Dict[str, Any]:
        """获取交易配置"""
        return self.get_section("trading")

    @property
    def exchange(self) -> Dict[str, Any]:
        """获取交易所配置"""
        return self.get_section("exchange")

    @property
    def ai(self) -> Dict[str, Any]:
        """获取 AI 配置"""
        return self.get_section("ai")

    @property
    def position(self) -> Dict[str, Any]:
        """获取仓位配置"""
        return self.get_section("position")

    @property
    def risk(self) -> Dict[str, Any]:
        """获取风险配置"""
        return self.get_section("risk")

    @property
    def monitoring(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.get_section("monitoring")

    @property
    def notification(self) -> Dict[str, Any]:
        """获取通知配置"""
        return self.get_section("notification")

    @property
    def logging(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get_section("logging")


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config(key: str = None, default: Any = None) -> Any:
    """获取配置值

    Args:
        key: 配置键，如果为 None 则返回所有配置
        default: 默认值

    Returns:
        配置值
    """
    if key is None:
        return _config_manager.get_all()
    return _config_manager.get(key, default)


def load_config(
    config_file: str = "config.yaml",
    validate: bool = True,
    resolve_env: bool = True,
    watch: bool = False,
) -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_file: 配置文件路径
        validate: 是否验证配置
        resolve_env: 是否解析环境变量
        watch: 是否监视文件变化

    Returns:
        配置字典
    """
    return _config_manager.load(
        config_file=config_file,
        validate=validate,
        resolve_env=resolve_env,
        watch=watch,
    )