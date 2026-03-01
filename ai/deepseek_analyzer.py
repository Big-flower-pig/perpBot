"""
PerpBot DeepSeek AI 分析模块

提供基于 DeepSeek API 的交易决策分析功能
- 决策记忆系统: 学习历史决策的成功/失败模式
- 自适应优化: 根据历史表现调整决策策略
- 上下文增强: 在提示词中注入历史经验
"""

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import statistics

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
    # 新增: 决策上下文
    market_regime: str = "UNKNOWN"  # TRENDING/RANGING/TRANSITIONAL
    entry_price: Optional[float] = None  # 决策时的价格

    @property
    def is_actionable(self) -> bool:
        """是否可执行的决策（非HOLD）"""
        return self.decision != Decision.HOLD

    @property
    def is_high_confidence(self) -> bool:
        """是否高信心"""
        return self.confidence == Confidence.HIGH


@dataclass
class DecisionOutcome:
    """决策结果跟踪"""

    decision_id: str
    decision: Decision
    confidence: Confidence
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_time_minutes: int = 0
    success: Optional[bool] = None  # True=盈利, False=亏损
    market_regime: str = "UNKNOWN"
    key_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None


@dataclass
class DecisionMemory:
    """决策记忆系统

    存储和分析历史决策，提取成功/失败模式
    """

    # 按决策类型统计
    decision_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"total": 0, "wins": 0, "losses": 0}
        )
    )

    # 按市场状态统计
    regime_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"total": 0, "wins": 0, "losses": 0}
        )
    )

    # 按信心程度统计
    confidence_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"total": 0, "wins": 0, "losses": 0}
        )
    )

    # 成功模式
    success_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # 失败模式
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # 最近决策结果
    recent_outcomes: List[DecisionOutcome] = field(default_factory=list)

    # 最大存储
    max_outcomes: int = 200

    def add_outcome(self, outcome: DecisionOutcome):
        """添加决策结果"""
        self.recent_outcomes.append(outcome)

        # 更新统计
        if outcome.success is not None:
            key = outcome.decision.value
            self.decision_stats[key]["total"] += 1
            if outcome.success:
                self.decision_stats[key]["wins"] += 1
            else:
                self.decision_stats[key]["losses"] += 1

            # 市场状态统计
            regime_key = outcome.market_regime
            self.regime_stats[regime_key]["total"] += 1
            if outcome.success:
                self.regime_stats[regime_key]["wins"] += 1
            else:
                self.regime_stats[regime_key]["losses"] += 1

            # 信心程度统计
            conf_key = outcome.confidence.value
            self.confidence_stats[conf_key]["total"] += 1
            if outcome.success:
                self.confidence_stats[conf_key]["wins"] += 1
            else:
                self.confidence_stats[conf_key]["losses"] += 1

            # 提取模式
            self._extract_pattern(outcome)

        # 清理旧数据
        if len(self.recent_outcomes) > self.max_outcomes:
            self.recent_outcomes = self.recent_outcomes[-self.max_outcomes // 2 :]

    def _extract_pattern(self, outcome: DecisionOutcome):
        """提取成功/失败模式"""
        pattern = {
            "decision": outcome.decision.value,
            "confidence": outcome.confidence.value,
            "market_regime": outcome.market_regime,
            "key_factors": outcome.key_factors[:3],  # 只保留前3个关键因素
            "pnl_percent": outcome.pnl_percent,
        }

        if outcome.success:
            if len(self.success_patterns) < 50:
                self.success_patterns.append(pattern)
        else:
            if len(self.failure_patterns) < 50:
                self.failure_patterns.append(pattern)

    def get_win_rate(self, decision_type: str = None) -> float:
        """获取胜率"""
        if decision_type:
            stats = self.decision_stats.get(decision_type, {})
            total = stats.get("total", 0)
            wins = stats.get("wins", 0)
        else:
            total = sum(s.get("total", 0) for s in self.decision_stats.values())
            wins = sum(s.get("wins", 0) for s in self.decision_stats.values())

        return (wins / total * 100) if total > 0 else 0

    def get_best_regime(self) -> Tuple[str, float]:
        """获取最佳市场状态"""
        best_regime = "UNKNOWN"
        best_rate = 0

        for regime, stats in self.regime_stats.items():
            total = stats.get("total", 0)
            wins = stats.get("wins", 0)
            if total >= 5:  # 至少5次交易
                rate = wins / total
                if rate > best_rate:
                    best_rate = rate
                    best_regime = regime

        return best_regime, best_rate * 100

    def get_confidence_reliability(self, confidence: str) -> float:
        """获取信心程度的可靠性"""
        stats = self.confidence_stats.get(confidence, {})
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        return (wins / total * 100) if total > 0 else 0

    def get_lessons_learned(self, limit: int = 5) -> List[str]:
        """获取经验教训"""
        lessons = []

        # 从成功模式中提取
        if self.success_patterns:
            recent_success = self.success_patterns[-3:]
            for p in recent_success:
                factors = ", ".join(p.get("key_factors", [])[:2])
                lessons.append(
                    f"成功: {p['decision']} 在 {p['market_regime']} 市场, 关键因素: {factors}"
                )

        # 从失败模式中提取
        if self.failure_patterns:
            recent_failure = self.failure_patterns[-3:]
            for p in recent_failure:
                factors = ", ".join(p.get("key_factors", [])[:2])
                lessons.append(
                    f"失败: {p['decision']} 在 {p['market_regime']} 市场, 关键因素: {factors}"
                )

        return lessons[:limit]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "decision_stats": dict(self.decision_stats),
            "regime_stats": dict(self.regime_stats),
            "confidence_stats": dict(self.confidence_stats),
            "total_outcomes": len(self.recent_outcomes),
            "overall_win_rate": self.get_win_rate(),
            "best_regime": self.get_best_regime(),
        }


class DeepSeekAnalyzer:
    """DeepSeek AI 分析器

    使用 DeepSeek API 进行交易决策分析

    核心功能:
    1. 智能决策分析
    2. 决策记忆系统 - 学习历史成功/失败模式
    3. 自适应优化 - 根据历史表现调整策略
    4. 上下文增强 - 在提示词中注入历史经验

    Example:
        analyzer = DeepSeekAnalyzer()

        # 分析交易决策
        decision = analyzer.analyze(market_data, indicators, position)

        if decision.is_actionable:
            print(f"AI建议: {decision.decision.value}")

        # 记录决策结果
        analyzer.record_outcome(decision_id, pnl, exit_price)
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

        # ===== 新增: 决策记忆系统 =====
        self._memory = DecisionMemory()
        self._pending_decisions: Dict[str, AIDecision] = {}  # 待验证结果的决策

        # 自适应参数
        self._adaptive_confidence_threshold = {
            Confidence.HIGH: 0.7,  # 高信心需要的胜率阈值
            Confidence.MEDIUM: 0.5,
            Confidence.LOW: 0.3,
        }

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

        # 提取市场状态
        market_regime = "UNKNOWN"
        if hasattr(indicators, "market_regime"):
            market_regime = indicators.market_regime
        elif isinstance(indicators, dict):
            market_regime = indicators.get("market_regime", "UNKNOWN")

        # 创建提示词上下文
        context = create_prompt_context(
            symbol=symbol,
            market_data=market_data,
            indicators=indicators,
            position=position,
            sentiment=sentiment,
        )

        # ===== 新增: 注入历史经验 =====
        memory_context = self._build_memory_context(market_regime)

        # 构建增强提示词
        prompt = PromptTemplates.build_enhanced_trading_prompt(context, memory_context)

        # 调用 API
        response_text, response_time = self._call_api(prompt)

        # 解析响应
        decision = self._parse_response(response_text)
        decision.raw_response = response_text
        decision.response_time_ms = response_time
        decision.market_regime = market_regime
        decision.entry_price = market_data.get("current_price")

        # ===== 新增: 自适应信心调整 =====
        decision = self._adjust_confidence(decision)

        # 记录历史
        self._add_to_history(decision)

        # 生成决策ID并存储待验证
        decision_id = (
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{decision.decision.value}"
        )
        self._pending_decisions[decision_id] = decision

        # 记录日志
        self._logger.info(
            f"[AI] DeepSeek 决策: {decision.decision.value} "
            f"(信心: {decision.confidence.value}, "
            f"风险: {decision.risk_level}, "
            f"市场: {market_regime}, "
            f"耗时: {response_time}ms)"
        )

        return decision

    def _build_memory_context(self, current_regime: str) -> Dict[str, Any]:
        """构建记忆上下文

        Args:
            current_regime: 当前市场状态

        Returns:
            包含历史经验的上下文
        """
        memory = self._memory

        # 获取经验教训
        lessons = memory.get_lessons_learned(limit=3)

        # 获取各类型决策的胜率
        buy_win_rate = memory.get_win_rate("BUY")
        sell_win_rate = memory.get_win_rate("SELL")

        # 获取最佳市场状态
        best_regime, best_rate = memory.get_best_regime()

        # 获取信心可靠性
        high_conf_reliability = memory.get_confidence_reliability("HIGH")
        medium_conf_reliability = memory.get_confidence_reliability("MEDIUM")

        # 当前市场状态的历史表现
        regime_stats = memory.regime_stats.get(current_regime, {})
        regime_total = regime_stats.get("total", 0)
        regime_wins = regime_stats.get("wins", 0)
        regime_win_rate = (regime_wins / regime_total * 100) if regime_total > 0 else 0

        return {
            "lessons_learned": lessons,
            "buy_win_rate": buy_win_rate,
            "sell_win_rate": sell_win_rate,
            "best_regime": best_regime,
            "best_regime_rate": best_rate,
            "high_conf_reliability": high_conf_reliability,
            "medium_conf_reliability": medium_conf_reliability,
            "current_regime": current_regime,
            "current_regime_win_rate": regime_win_rate,
            "current_regime_trades": regime_total,
            "total_trades": len(memory.recent_outcomes),
        }

    def _adjust_confidence(self, decision: AIDecision) -> AIDecision:
        """根据历史表现自适应调整信心程度

        Args:
            decision: 原始决策

        Returns:
            调整后的决策
        """
        memory = self._memory

        # 获取该信心水平的历史可靠性
        reliability = memory.get_confidence_reliability(decision.confidence.value)

        # 获取该决策类型的历史胜率
        decision_win_rate = memory.get_win_rate(decision.decision.value)

        # 调整逻辑
        if decision.confidence == Confidence.HIGH:
            # 高信心需要历史验证
            if reliability < 50 and len(memory.recent_outcomes) >= 10:
                # 高信心历史表现不佳，降级
                decision.confidence = Confidence.MEDIUM
                self._logger.debug("高信心降级: 历史可靠性不足")

        elif decision.confidence == Confidence.MEDIUM:
            # 中信心检查
            if reliability < 40 and len(memory.recent_outcomes) >= 10:
                decision.confidence = Confidence.LOW
                self._logger.debug("中信心降级: 历史可靠性不足")

        # 如果该决策类型历史胜率很低，添加警告
        if decision_win_rate < 35 and decision_win_rate > 0:
            decision.risk_level = "HIGH"
            self._logger.debug(
                f"风险升级: {decision.decision.value} 历史胜率低 ({decision_win_rate:.1f}%)"
            )

        return decision

    def record_outcome(
        self,
        decision_id: str,
        pnl: float,
        exit_price: float,
        holding_time_minutes: int = 0,
    ) -> bool:
        """记录决策结果

        在交易结束后调用，用于学习反馈

        Args:
            decision_id: 决策ID
            pnl: 盈亏金额
            exit_price: 平仓价格
            holding_time_minutes: 持仓时间(分钟)

        Returns:
            是否成功记录
        """
        if decision_id not in self._pending_decisions:
            self._logger.warning(f"未找到决策ID: {decision_id}")
            return False

        decision = self._pending_decisions[decision_id]

        # 计算盈亏比例
        pnl_percent = 0
        if decision.entry_price and decision.entry_price > 0:
            pnl_percent = (pnl / decision.entry_price) * 100

        # 创建结果记录
        outcome = DecisionOutcome(
            decision_id=decision_id,
            decision=decision.decision,
            confidence=decision.confidence,
            entry_price=decision.entry_price or 0,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            holding_time_minutes=holding_time_minutes,
            success=pnl > 0,
            market_regime=decision.market_regime,
            key_factors=decision.key_factors,
            closed_at=datetime.now(),
        )

        # 添加到记忆系统
        self._memory.add_outcome(outcome)

        # 清理待验证列表
        del self._pending_decisions[decision_id]

        self._logger.info(
            f"[AI] 决策结果记录: {decision_id} "
            f"PnL={pnl:+.2f} ({pnl_percent:+.2f}%) "
            f"结果={'盈利' if pnl > 0 else '亏损'}"
        )

        return True

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计"""
        return self._memory.to_dict()

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
        """获取统计信息 (增强版)"""
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

            # 合并记忆系统统计
            memory_stats = self._memory.to_dict()

            return {
                "total_requests": self._request_count,
                "average_response_time_ms": avg_response_time,
                "decision_distribution": decision_counts,
                "history_size": len(self._decision_history),
                # 新增: 记忆系统统计
                "memory": memory_stats,
                "pending_decisions": len(self._pending_decisions),
            }

    def get_recent_decisions(self, count: int = 10) -> List[AIDecision]:
        """获取最近的决策历史"""
        with self._lock:
            return self._decision_history[-count:]

    def get_performance_summary(self) -> str:
        """获取性能摘要文本"""
        stats = self.get_stats()
        memory = stats.get("memory", {})

        return (
            f"AI性能摘要:\n"
            f"  总请求数: {stats['total_requests']}\n"
            f"  平均响应时间: {stats['average_response_time_ms']:.0f}ms\n"
            f"  总体胜率: {memory.get('overall_win_rate', 0):.1f}%\n"
            f"  最佳市场状态: {memory.get('best_regime', 'N/A')} "
            f"({memory.get('best_regime_rate', 0):.1f}%)"
        )


# 全局分析器实例
_analyzer: Optional[DeepSeekAnalyzer] = None


def get_analyzer() -> DeepSeekAnalyzer:
    """获取全局分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DeepSeekAnalyzer()
    return _analyzer
