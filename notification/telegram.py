"""
PerpBot Telegram 通知模块

提供 Telegram 消息推送功能：
- 交易通知
- 警报通知
- 状态报告
- 错误通知
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

import requests

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.decorators import retry_on_failure


class MessageType(Enum):
    """消息类型"""
    INFO = "info"
    TRADE = "trade"
    ALERT = "alert"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    REPORT = "report"


@dataclass
class TelegramMessage:
    """Telegram 消息"""
    text: str
    message_type: MessageType = MessageType.INFO
    parse_mode: str = "Markdown"
    disable_notification: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class TelegramNotifier:
    """Telegram 通知器

    发送各类通知到 Telegram

    Example:
        notifier = TelegramNotifier()

        # 发送交易通知
        notifier.send_trade(
            action="BUY",
            symbol="BTC/USDT",
            price=95000,
            size=0.1
        )

        # 发送警报
        notifier.send_alert("价格异常波动!")
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self._logger = get_logger("telegram")
        self._token = get_config("notification.telegram.bot_token")
        self._chat_id = get_config("notification.telegram.chat_id")
        self._enabled = get_config("notification.telegram.enabled", False)

        # 消息队列（用于批量发送）
        self._message_queue: List[TelegramMessage] = []
        self._lock = threading.Lock()

        # 速率限制
        self._last_send_time = 0
        self._min_interval = 1.0  # 最小发送间隔（秒）

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self._token and self._chat_id)

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled and self.is_configured()

    @retry_on_failure(max_retries=3, delay=2.0)
    def send(self, message: TelegramMessage) -> bool:
        """发送消息

        Args:
            message: 消息对象

        Returns:
            是否发送成功
        """
        if not self.is_enabled():
            self._logger.debug("Telegram 通知未启用或未配置")
            return False

        # 格式化消息
        formatted_text = self._format_message(message)

        # 发送请求
        return self._send_request(formatted_text, message.parse_mode, message.disable_notification)

    def send_text(self, text: str, message_type: MessageType = MessageType.INFO) -> bool:
        """发送纯文本消息

        Args:
            text: 文本内容
            message_type: 消息类型

        Returns:
            是否发送成功
        """
        message = TelegramMessage(text=text, message_type=message_type)
        return self.send(message)

    def send_trade(
        self,
        action: str,
        symbol: str,
        price: float,
        size: float,
        pnl: Optional[float] = None,
        confidence: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """发送交易通知

        Args:
            action: 操作 (BUY/SELL/CLOSE)
            symbol: 交易对
            price: 价格
            size: 数量
            pnl: 盈亏
            confidence: 信心程度
            reason: 原因

        Returns:
            是否发送成功
        """
        # 构建消息
        lines = [
            f"📊 *交易信号执行*",
            f"",
            f"🎯 *操作*: {action}",
            f"💱 *交易对*: `{symbol}`",
            f"💰 *价格*: `{price:.4f}`",
            f"📐 *数量*: `{size}`",
        ]

        if pnl is not None:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"{pnl_emoji} *盈亏*: `{pnl:+.2f} USDT`")

        if confidence:
            lines.append(f"⚡ *信心*: `{confidence}`")

        if reason:
            lines.append(f"📝 *原因*: {reason}")

        lines.append(f"")
        lines.append(f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.TRADE,
        )

        return self.send(message)

    def send_alert(
        self,
        title: str,
        content: str,
        severity: str = "MEDIUM",
    ) -> bool:
        """发送警报通知

        Args:
            title: 标题
            content: 内容
            severity: 严重程度

        Returns:
            是否发送成功
        """
        # 根据严重程度选择 emoji
        severity_emojis = {
            "LOW": "⚠️",
            "MEDIUM": "🔶",
            "HIGH": "🔴",
            "CRITICAL": "🚨",
        }
        emoji = severity_emojis.get(severity, "⚠️")

        lines = [
            f"{emoji} *{title}*",
            f"",
            f"{content}",
            f"",
            f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        ]

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.ALERT,
        )

        return self.send(message)

    def send_error(
        self,
        error_type: str,
        error_message: str,
        details: Optional[str] = None,
    ) -> bool:
        """发送错误通知

        Args:
            error_type: 错误类型
            error_message: 错误消息
            details: 详细信息

        Returns:
            是否发送成功
        """
        lines = [
            f"❌ *错误报告*",
            f"",
            f"🔴 *类型*: `{error_type}`",
            f"💬 *消息*: {error_message}",
        ]

        if details:
            lines.append(f"📋 *详情*: `{details[:200]}`")

        lines.append(f"")
        lines.append(f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.ERROR,
        )

        return self.send(message)

    def send_report(
        self,
        title: str,
        stats: Dict[str, Any],
    ) -> bool:
        """发送统计报告

        Args:
            title: 标题
            stats: 统计数据

        Returns:
            是否发送成功
        """
        lines = [
            f"📈 *{title}*",
            f"",
        ]

        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"• *{key}*: `{value:.4f}`")
            else:
                lines.append(f"• *{key}*: `{value}`")

        lines.append(f"")
        lines.append(f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.REPORT,
        )

        return self.send(message)

    def send_startup(self, version: str = "1.0.0") -> bool:
        """发送启动通知

        Args:
            version: 版本号

        Returns:
            是否发送成功
        """
        lines = [
            f"🚀 *PerpBot 启动成功*",
            f"",
            f"📌 *版本*: `{version}`",
            f"💱 *交易对*: `{get_config('trading.symbol', 'N/A')}`",
            f"⚙️ *杠杆*: `{get_config('trading.leverage', 'N/A')}x`",
            f"",
            f"✅ 系统已就绪，开始监控市场...",
            f"",
            f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        ]

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.SUCCESS,
        )

        return self.send(message)

    def send_shutdown(self, reason: str = "正常关闭") -> bool:
        """发送关闭通知

        Args:
            reason: 关闭原因

        Returns:
            是否发送成功
        """
        lines = [
            f"🛑 *PerpBot 已停止*",
            f"",
            f"📝 *原因*: {reason}",
            f"",
            f"⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        ]

        message = TelegramMessage(
            text="\n".join(lines),
            message_type=MessageType.WARNING,
        )

        return self.send(message)

    def _format_message(self, message: TelegramMessage) -> str:
        """格式化消息

        Args:
            message: 消息对象

        Returns:
            格式化后的文本
        """
        # 添加类型前缀
        type_prefixes = {
            MessageType.INFO: "ℹ️",
            MessageType.TRADE: "📊",
            MessageType.ALERT: "⚠️",
            MessageType.ERROR: "❌",
            MessageType.SUCCESS: "✅",
            MessageType.WARNING: "🔶",
            MessageType.REPORT: "📈",
        }

        prefix = type_prefixes.get(message.message_type, "ℹ️")

        return f"{prefix} {message.text}"

    def _send_request(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
    ) -> bool:
        """发送 HTTP 请求

        Args:
            text: 消息文本
            parse_mode: 解析模式
            disable_notification: 是否静音

        Returns:
            是否成功
        """
        import time

        # 速率限制
        current_time = time.time()
        elapsed = current_time - self._last_send_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            url = self.API_URL.format(token=self._token)

            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            self._last_send_time = time.time()

            self._logger.debug(f"Telegram 消息发送成功")
            return True

        except requests.exceptions.RequestException as e:
            self._logger.error(f"Telegram 消息发送失败: {e}")
            return False


# 全局通知器实例
_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    """获取全局 Telegram 通知器实例"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier