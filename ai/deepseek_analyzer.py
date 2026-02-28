"""
PerpBot DeepSeek AI 分析模块

提供基于 DeepSeek API 的交易决策分析功能
"""

import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

import requests

from utils.logger import get_logger, TradingLogger
from utils.config import get_config
from utils.decorators import retry_on_failure, rate_limit
from ai.prompt_templates import PromptTemplates, PromptContext, create_prompt_context


class Decision(Enum):
    """交易决策枚举"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class Confidence(Enum):
    """信心程度枚举"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class AIDecision:
    """AI 决策结果"""

    decision: Decision
    confidence: Confidence
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_level: str = "MEDIUM"
    key_factors: List[str] = field(default_factory=list)
    raw_response: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: int = 0

    @property
    def is_actionable(self) -> bool:
        """是否可执行的决策（非HOLD）"""
        return self.decision != Decision.HOLD

    @property
    def is_high_confidence(self) -> bool:
        """是否高信心"""
        return self.confidence == Confidence.HIGH


class DeepSeekAnalyzer:
    """DeepSeek AI 分析器

    使用 DeepSeek API 进行交易决策分析

    Example:
        analyzer = DeepSeekAnalyzer()

        # 分析交易决策
        decision = analyzer.analyze(market_data, indicators, position)

        if decision.is_actionable:
            print(f"AI建议: {decision.decision.value}")
    """

    # DeepSeek API 配置
    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self):
        self._logger = get_logger("ai")
        self._api_key = get_config("ai.api_key")
        self._model = get_config("ai.model", "deepseek-chat")
        self._temperature = get_config("ai.temperature", 0.3)
        self._max_tokens = get_config("ai.max_tokens", 500)

        # 请求统计
        self._request_count = 0
        self._total_response_time = 0
        self._lock = threading.Lock()

        # 决策历史
        self._decision_history: List[AIDecision] = []
        self._max_history = 100

    @retry_on_failure(max_retries=3, delay=1.0, exponential_backoff=True)
    @rate_limit(calls_per_minute=20)
    def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        position: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
    ) -> AIDecision:
        """分析市场数据并给出交易决策

        Args:
            market_data: 市场数据
            indicators: 技术指标
            position: 当前持仓
            sentiment: 市场情绪

        Returns:
            AIDecision 决策结果
        """
        symbol = get_config("trading.symbol", "BTC/USDT:USDT")

        # 创建提示词上下文
        context = create_prompt_context(
            symbol=symbol,
            market_data=market_data,
            indicators=indicators,
            position=position,
            sentiment=sentiment,
        )

        # 构建提示词
        prompt = PromptTemplates.build_trading_prompt(context)

        # 调用 API
        response_text, response_time = self._call_api(prompt)

        # 解析响应
        decision = self._parse_response(response_text)
        decision.raw_response = response_text
        decision.response_time_ms = response_time

        # 记录历史
        self._add_to_history(decision)

        # 记录日志 - 使用 info 方法记录 AI 决策
        self._logger.info(
            f"[AI] DeepSeek 决策: {decision.decision.value} "
            f"(信心: {decision.confidence.value}, "
            f"风险: {decision.risk_level}, "
            f"耗时: {response_time}ms)"
        )

        return decision

    def analyze_close_decision(
        self,
        market_data: Dict[str, Any],
        position: Dict[str, Any],
    ) -> AIDecision:
        """分析是否应该平仓

        Args:
            market_data: 市场数据
            position: 当前持仓

        Returns:
            AIDecision 决策结果
        """
        symbol = get_config("trading.symbol", "BTC/USDT:USDT")

        context = PromptContext(
            symbol=symbol,
            current_price=market_data.get("current_price", 0),
            price_change_24h=market_data.get("change_24h", 0),
            volume_24h=market_data.get("volume_24h", 0),
            high_24h=market_data.get("high_24h", 0),
            low_24h=market_data.get("low_24h", 0),
            funding_rate=market_data.get("funding_rate"),
            rsi=market_data.get("rsi"),
            has_position=True,
            position_side=position.get("side"),
            position_size=position.get("size"),
            entry_price=position.get("entry_price"),
            unrealized_pnl=position.get("unrealized_pnl"),
        )

        prompt = PromptTemplates.build_close_position_prompt(context)

        response_text, response_time = self._call_api(prompt)
        decision = self._parse_response(response_text)
        decision.raw_response = response_text
        decision.response_time_ms = response_time

        return decision

    def analyze_risk_alert(
        self,
        market_data: Dict[str, Any],
        position: Dict[str, Any],
        alert_type: str,
        alert_message: str,
    ) -> AIDecision:
        """分析风险警报并给出决策

        Args:
            market_data: 市场数据
            position: 当前持仓
            alert_type: 警报类型
            alert_message: 警报消息

        Returns:
            AIDecision 决策结果
        """
        symbol = get_config("trading.symbol", "BTC/USDT:USDT")

        context = PromptContext(
            symbol=symbol,
            current_price=market_data.get("current_price", 0),
            price_change_24h=market_data.get("change_24h", 0),
            volume_24h=0,
            high_24h=0,
            low_24h=0,
            has_position=True,
            position_side=position.get("side"),
            position_size=position.get("size"),
            entry_price=position.get("entry_price"),
            unrealized_pnl=position.get("unrealized_pnl"),
        )

        prompt = PromptTemplates.build_risk_alert_prompt(
            context, alert_type, alert_message
        )

        response_text, response_time = self._call_api(prompt)
        decision = self._parse_response(response_text)
        decision.raw_response = response_text
        decision.response_time_ms = response_time

        return decision

    def _call_api(self, prompt: str) -> tuple[str, int]:
        """调用 DeepSeek API

        Args:
            prompt: 提示词

        Returns:
            (响应文本, 响应时间毫秒)
        """
        if not self._api_key:
            raise ValueError("DeepSeek API key not configured")

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": PromptTemplates.SYSTEM_ROLE},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            response_time = int((time.time() - start_time) * 1000)

            # 更新统计
            with self._lock:
                self._request_count += 1
                self._total_response_time += response_time

            return content, response_time

        except requests.exceptions.Timeout:
            self._logger.error("DeepSeek API 请求超时")
            raise
        except requests.exceptions.RequestException as e:
            self._logger.error(f"DeepSeek API 请求失败: {e}")
            raise

    def _parse_response(self, response_text: str) -> AIDecision:
        """解析 API 响应

        Args:
            response_text: API 响应文本

        Returns:
            AIDecision 对象
        """
        try:
            # 尝试提取 JSON
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)

            # 解析决策
            decision_str = data.get("decision", "HOLD").upper()
            decision = (
                Decision(decision_str)
                if decision_str in [d.value for d in Decision]
                else Decision.HOLD
            )

            # 解析信心
            confidence_str = data.get("confidence", "MEDIUM").upper()
            confidence = (
                Confidence(confidence_str)
                if confidence_str in [c.value for c in Confidence]
                else Confidence.MEDIUM
            )

            return AIDecision(
                decision=decision,
                confidence=confidence,
                reason=data.get("reason", ""),
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                risk_level=data.get("risk_level", "MEDIUM"),
                key_factors=data.get("key_factors", []),
            )

        except (json.JSONDecodeError, ValueError) as e:
            self._logger.warning(
                f"解析 AI 响应失败: {e}, 原始响应: {response_text[:200]}"
            )

            # 返回默认的 HOLD 决策
            return AIDecision(
                decision=Decision.HOLD,
                confidence=Confidence.LOW,
                reason=f"解析响应失败: {str(e)}",
            )

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON

        Args:
            text: 可能包含 JSON 的文本

        Returns:
            JSON 字符串
        """
        # 尝试直接解析
        text = text.strip()
        if text.startswith("{"):
            return text

        # 查找 JSON 块
        import re

        # 尝试匹配 ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json_match.group(1).strip()

        # 尝试匹配 { ... }
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            return brace_match.group(0)

        return text

    def _add_to_history(self, decision: AIDecision):
        """添加到决策历史"""
        with self._lock:
            self._decision_history.append(decision)
            if len(self._decision_history) > self._max_history:
                self._decision_history = self._decision_history[
                    -self._max_history // 2 :
                ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            avg_response_time = (
                self._total_response_time / self._request_count
                if self._request_count > 0
                else 0
            )

            # 统计决策分布
            decision_counts = {}
            for d in self._decision_history:
                key = d.decision.value
                decision_counts[key] = decision_counts.get(key, 0) + 1

            return {
                "total_requests": self._request_count,
                "average_response_time_ms": avg_response_time,
                "decision_distribution": decision_counts,
                "history_size": len(self._decision_history),
            }

    def get_recent_decisions(self, count: int = 10) -> List[AIDecision]:
        """获取最近的决策历史"""
        with self._lock:
            return self._decision_history[-count:]


# 全局分析器实例
_analyzer: Optional[DeepSeekAnalyzer] = None


def get_analyzer() -> DeepSeekAnalyzer:
    """获取全局分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DeepSeekAnalyzer()
    return _analyzer
