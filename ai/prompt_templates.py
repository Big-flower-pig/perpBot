"""
PerpBot AI 提示词模板模块

提供各种交易决策场景的提示词模板
- 基础模板: 标准交易决策
- 增强模板: 注入历史经验和记忆上下文
- 专业模板: 包含更多技术分析维度
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class PromptContext:
    """提示词上下文"""

    symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    # 技术指标
    ma_7: Optional[float] = None
    ma_25: Optional[float] = None
    ma_99: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[Dict] = None
    bollinger_bands: Optional[Dict] = None
    # 持仓信息
    has_position: bool = False
    position_side: Optional[str] = None
    position_size: Optional[float] = None
    entry_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    # 市场情绪
    fear_greed_index: Optional[int] = None
    market_sentiment: Optional[str] = None
    # 时间
    timestamp: str = ""


class PromptTemplates:
    """提示词模板集合"""

    # 系统角色定义 (增强版)
    SYSTEM_ROLE = """你是一个专业的加密货币合约交易助手，具备以下能力：
1. 深刻理解加密货币市场动态和价格行为
2. 熟练运用技术分析指标（MA、RSI、MACD、布林带等）
3. 理解合约交易机制（杠杆、保证金、资金费率等）
4. 具备风险管理意识，注重资金安全
5. 能够综合多维度信息做出理性判断

你的决策原则：
- 积极寻找交易机会，在技术信号明确时果断入场
- 顺势而为，优先跟随主趋势方向
- 严格执行止损止盈，每笔交易必须设置止损
- 保持冷静，基于数据而非情绪做出决策
- 当技术指标出现超买超卖反转信号时，可以逆势操作

开仓信号判断标准：
- BUY信号：价格在重要支撑位企稳、RSI超卖反弹、MACD金叉、突破均线压力
- SELL信号：价格在重要阻力位回落、RSI超买回落、MACD死叉、跌破均线支撑
- HOLD信号：仅在信号不明确或市场处于震荡区间时使用

你必须以JSON格式返回决策结果，格式如下：
{
    "decision": "BUY/SELL/HOLD/CLOSE",
    "confidence": "HIGH/MEDIUM/LOW",
    "reason": "决策理由的简短说明",
    "stop_loss": 止损价格(数字),
    "take_profit": 止盈价格(数字),
    "risk_level": "LOW/MEDIUM/HIGH",
    "key_factors": ["关键因素1", "关键因素2"]
}"""

    # 交易决策提示词模板
    TRADING_DECISION_TEMPLATE = """## 市场概况
- 交易对: {symbol}
- 当前价格: {current_price}
- 24小时涨跌: {price_change_24h:+.2f}%
- 24小时最高: {high_24h}
- 24小时最低: {low_24h}
- 24小时成交量: {volume_24h:,.0f}

## 技术指标
{technical_indicators}

## 市场情绪
{market_sentiment}

## 当前持仓
{position_info}

## 分析要求
请基于以上信息，给出你的交易决策。注意：
1. 优先寻找开仓机会，技术信号明确时果断入场
2. 综合考虑技术面、情绪面和持仓情况
3. 如果有持仓，评估是否需要加仓或平仓
4. 每笔交易必须设置合理的止损止盈价位
5. 当出现以下信号时积极开仓：
   - BUY: RSI<30超卖、MACD金叉、价格触及布林带下轨、突破均线压力
   - SELL: RSI>70超买、MACD死叉、价格触及布林带上轨、跌破均线支撑

请返回JSON格式的决策结果："""

    # ===== 新增: 增强版交易决策模板 =====
    ENHANCED_TRADING_TEMPLATE = """## 市场概况
- 交易对: {symbol}
- 当前价格: {current_price}
- 24小时涨跌: {price_change_24h:+.2f}%
- 24小时最高: {high_24h}
- 24小时最低: {low_24h}
- 24小时成交量: {volume_24h:,.0f}

## 技术指标
{technical_indicators}

## 市场情绪
{market_sentiment}

## 当前持仓
{position_info}

## 历史经验 (重要参考)
{memory_context}

## 分析要求
请基于以上信息和历史经验，给出你的交易决策。注意：
1. 参考历史经验中的成功/失败模式
2. 在历史表现较好的市场状态下更积极
3. 如果当前市场状态历史胜率低，提高警惕
4. 综合考虑技术面、情绪面和持仓情况
5. 每笔交易必须设置合理的止损止盈价位

请返回JSON格式的决策结果："""

    # 技术指标格式化模板
    TECHNICAL_INDICATORS_TEMPLATE = """### 均线系统
- MA7: {ma_7} (短期趋势)
- MA25: {ma_25} (中期趋势)
- MA99: {ma_99} (长期趋势)

### 动量指标
- RSI(14): {rsi}

### MACD
- DIF: {macd_dif}
- DEA: {macd_dea}
- 柱状图: {macd_histogram}

### 布林带
- 上轨: {bb_upper}
- 中轨: {bb_middle}
- 下轨: {bb_lower}
- 当前位置: {bb_position}"""

    # 市场情绪格式化模板
    MARKET_SENTIMENT_TEMPLATE = """### 情绪指标
- 恐惧贪婪指数: {fear_greed_index}/100
- 市场情绪: {market_sentiment}

### 链上数据
- 资金费率: {funding_rate}
- 持仓量: {open_interest}"""

    # 持仓信息格式化模板
    POSITION_INFO_TEMPLATE = """### 当前持仓
- 方向: {position_side}
- 数量: {position_size}
- 入场价: {entry_price}
- 未实现盈亏: {unrealized_pnl:+.2f} USDT
- 盈亏比例: {pnl_percent:+.2f}%"""

    NO_POSITION_TEMPLATE = """### 当前持仓
- 无持仓"""

    @classmethod
    def build_trading_prompt(cls, context: PromptContext) -> str:
        """构建交易决策提示词

        Args:
            context: 提示词上下文

        Returns:
            完整的提示词
        """
        # 格式化技术指标
        technical_indicators = cls._format_technical_indicators(context)

        # 格式化市场情绪
        market_sentiment = cls._format_market_sentiment(context)

        # 格式化持仓信息
        position_info = cls._format_position_info(context)

        # 构建完整提示词
        prompt = cls.TRADING_DECISION_TEMPLATE.format(
            symbol=context.symbol,
            current_price=context.current_price,
            price_change_24h=context.price_change_24h,
            high_24h=context.high_24h,
            low_24h=context.low_24h,
            volume_24h=context.volume_24h,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            position_info=position_info,
        )

        return prompt

    @classmethod
    def build_enhanced_trading_prompt(
        cls, context: PromptContext, memory_context: Dict[str, Any]
    ) -> str:
        """构建增强版交易决策提示词

        注入历史经验和记忆上下文，帮助AI做出更明智的决策

        Args:
            context: 提示词上下文
            memory_context: 记忆上下文 (包含历史经验)

        Returns:
            完整的增强版提示词
        """
        # 格式化技术指标
        technical_indicators = cls._format_technical_indicators(context)

        # 格式化市场情绪
        market_sentiment = cls._format_market_sentiment(context)

        # 格式化持仓信息
        position_info = cls._format_position_info(context)

        # 格式化记忆上下文
        memory_text = cls._format_memory_context(memory_context)

        # 构建完整提示词
        prompt = cls.ENHANCED_TRADING_TEMPLATE.format(
            symbol=context.symbol,
            current_price=context.current_price,
            price_change_24h=context.price_change_24h,
            high_24h=context.high_24h,
            low_24h=context.low_24h,
            volume_24h=context.volume_24h,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            position_info=position_info,
            memory_context=memory_text,
        )

        return prompt

    @classmethod
    def _format_memory_context(cls, memory: Dict[str, Any]) -> str:
        """格式化记忆上下文

        Args:
            memory: 记忆数据

        Returns:
            格式化的文本
        """
        lines = []

        # 历史胜率
        total_trades = memory.get("total_trades", 0)
        if total_trades > 0:
            buy_rate = memory.get("buy_win_rate", 0)
            sell_rate = memory.get("sell_win_rate", 0)

            lines.append(f"### 历史表现 ({total_trades}笔交易)")
            lines.append(f"- BUY 决策胜率: {buy_rate:.1f}%")
            lines.append(f"- SELL 决策胜率: {sell_rate:.1f}%")

            # 最佳市场状态
            best_regime = memory.get("best_regime", "N/A")
            best_rate = memory.get("best_regime_rate", 0)
            if best_regime != "UNKNOWN":
                lines.append(f"- 最佳表现市场: {best_regime} ({best_rate:.1f}%)")

            # 当前市场状态表现
            current_regime = memory.get("current_regime", "UNKNOWN")
            current_rate = memory.get("current_regime_win_rate", 0)
            current_trades = memory.get("current_regime_trades", 0)
            if current_trades > 0:
                lines.append(
                    f"- 当前市场({current_regime})胜率: {current_rate:.1f}% ({current_trades}笔)"
                )

        # 经验教训
        lessons = memory.get("lessons_learned", [])
        if lessons:
            lines.append("\n### 近期经验教训")
            for lesson in lessons[:5]:
                lines.append(f"- {lesson}")

        # 信心可靠性
        high_rel = memory.get("high_conf_reliability", 0)
        med_rel = memory.get("medium_conf_reliability", 0)
        if high_rel > 0 or med_rel > 0:
            lines.append("\n### 信心可靠性")
            if high_rel > 0:
                lines.append(f"- HIGH信心实际胜率: {high_rel:.1f}%")
            if med_rel > 0:
                lines.append(f"- MEDIUM信心实际胜率: {med_rel:.1f}%")

        if not lines:
            return "暂无历史数据"

        return "\n".join(lines)

    @classmethod
    def _format_technical_indicators(cls, context: PromptContext) -> str:
        """格式化技术指标"""
        # 计算 MACD 信息
        macd_dif = context.macd.get("dif", "N/A") if context.macd else "N/A"
        macd_dea = context.macd.get("dea", "N/A") if context.macd else "N/A"
        macd_hist = context.macd.get("histogram", "N/A") if context.macd else "N/A"

        # 计算布林带位置
        bb_position = "N/A"
        if context.bollinger_bands and context.current_price:
            upper = context.bollinger_bands.get("upper", 0)
            lower = context.bollinger_bands.get("lower", 0)
            middle = context.bollinger_bands.get("middle", 0)
            if upper and lower:
                # 计算价格在布林带中的位置 (0-100%)
                bb_position = (
                    f"{((context.current_price - lower) / (upper - lower) * 100):.1f}%"
                )

        return cls.TECHNICAL_INDICATORS_TEMPLATE.format(
            ma_7=context.ma_7 or "N/A",
            ma_25=context.ma_25 or "N/A",
            ma_99=context.ma_99 or "N/A",
            rsi=context.rsi or "N/A",
            macd_dif=macd_dif,
            macd_dea=macd_dea,
            macd_histogram=macd_hist,
            bb_upper=context.bollinger_bands.get("upper", "N/A")
            if context.bollinger_bands
            else "N/A",
            bb_middle=context.bollinger_bands.get("middle", "N/A")
            if context.bollinger_bands
            else "N/A",
            bb_lower=context.bollinger_bands.get("lower", "N/A")
            if context.bollinger_bands
            else "N/A",
            bb_position=bb_position,
        )

    @classmethod
    def _format_market_sentiment(cls, context: PromptContext) -> str:
        """格式化市场情绪"""
        return cls.MARKET_SENTIMENT_TEMPLATE.format(
            fear_greed_index=context.fear_greed_index or "N/A",
            market_sentiment=context.market_sentiment or "N/A",
            funding_rate=context.funding_rate or "N/A",
            open_interest=context.open_interest or "N/A",
        )

    @classmethod
    def _format_position_info(cls, context: PromptContext) -> str:
        """格式化持仓信息"""
        if not context.has_position:
            return cls.NO_POSITION_TEMPLATE

        # 计算盈亏比例
        pnl_percent = 0
        if context.entry_price and context.unrealized_pnl and context.position_size:
            # 简化计算
            pnl_percent = (
                context.unrealized_pnl / (context.entry_price * context.position_size)
            ) * 100

        return cls.POSITION_INFO_TEMPLATE.format(
            position_side="做多" if context.position_side == "long" else "做空",
            position_size=context.position_size,
            entry_price=context.entry_price,
            unrealized_pnl=context.unrealized_pnl or 0,
            pnl_percent=pnl_percent,
        )

    @classmethod
    def build_close_position_prompt(cls, context: PromptContext) -> str:
        """构建平仓决策提示词

        Args:
            context: 提示词上下文

        Returns:
            完整的提示词
        """
        # 计算盈亏比例
        pnl_percent = 0
        if context.entry_price and context.current_price:
            if context.position_side == "long":
                pnl_percent = (
                    (context.current_price - context.entry_price) / context.entry_price
                ) * 100
            else:
                pnl_percent = (
                    (context.entry_price - context.current_price) / context.entry_price
                ) * 100

        prompt = f"""## 紧急平仓评估

### 当前持仓
- 方向: {"做多" if context.position_side == "long" else "做空"}
- 入场价: {context.entry_price}
- 当前价: {context.current_price}
- 未实现盈亏: {context.unrealized_pnl:+.2f} USDT
- 盈亏比例: {pnl_percent:+.2f}%

### 市场状况
- 24小时涨跌: {context.price_change_24h:+.2f}%
- RSI: {context.rsi}
- 资金费率: {context.funding_rate}

### 评估要求
请评估是否需要立即平仓：
1. 如果亏损接近止损(-2%以上)，建议CLOSE
2. 如果盈利达到目标(+3%以上)，建议CLOSE
3. 如果市场出现反转信号，建议CLOSE
4. 如果RSI极端(>80或<20)，考虑CLOSE
5. 否则建议HOLD继续持有

请返回JSON格式的决策结果："""
        return prompt

    @classmethod
    def build_risk_alert_prompt(
        cls, context: PromptContext, alert_type: str, alert_message: str
    ) -> str:
        """构建风险警报提示词

        Args:
            context: 提示词上下文
            alert_type: 警报类型
            alert_message: 警报消息

        Returns:
            完整的提示词
        """
        prompt = f"""## 风险警报

### 警报类型: {alert_type}
### 警报详情: {alert_message}

### 当前持仓
- 方向: {"做多" if context.position_side == "long" else "做空"}
- 入场价: {context.entry_price}
- 当前价: {context.current_price}
- 未实现盈亏: {context.unrealized_pnl:+.2f} USDT

### 紧急决策
请立即评估风险并给出决策：
- CLOSE: 立即平仓止损
- HOLD: 继续持有但提高警惕

请返回JSON格式的决策结果："""
        return prompt


def create_prompt_context(
    symbol: str,
    market_data: Dict[str, Any],
    indicators,
    position: Optional[Dict[str, Any]] = None,
    sentiment: Optional[Dict[str, Any]] = None,
) -> PromptContext:
    """创建提示词上下文的便捷函数

    Args:
        symbol: 交易对
        market_data: 市场数据
        indicators: 技术指标（可以是字典或 TechnicalIndicators 对象）
        position: 持仓信息
        sentiment: 市场情绪

    Returns:
        PromptContext 实例
    """
    has_position = position is not None and position.get("size", 0) > 0

    # 处理 indicators 参数 - 支持字典和 dataclass 对象
    if hasattr(indicators, "__dataclass_fields__"):
        # 是 dataclass 对象，转换为字典
        from dataclasses import asdict

        indicators_dict = asdict(indicators)
    elif hasattr(indicators, "get"):
        # 是字典
        indicators_dict = indicators
    else:
        # 其他类型，使用空字典
        indicators_dict = {}

    # 构建布林带字典
    bollinger_bands = {
        "upper": indicators_dict.get("bb_upper"),
        "middle": indicators_dict.get("bb_middle"),
        "lower": indicators_dict.get("bb_lower"),
    }

    # 构建 MACD 字典
    macd = {
        "dif": indicators_dict.get("macd"),
        "dea": indicators_dict.get("macd_signal"),
        "histogram": indicators_dict.get("macd_histogram"),
    }

    return PromptContext(
        symbol=symbol,
        current_price=market_data.get("current_price", 0),
        price_change_24h=market_data.get("change_24h", 0),
        volume_24h=market_data.get("volume_24h", 0),
        high_24h=market_data.get("high_24h", 0),
        low_24h=market_data.get("low_24h", 0),
        funding_rate=market_data.get("funding_rate"),
        open_interest=market_data.get("open_interest"),
        ma_7=indicators_dict.get("sma_5"),  # 使用 sma_5 作为 ma_7
        ma_25=indicators_dict.get("sma_20"),  # 使用 sma_20 作为 ma_25
        ma_99=indicators_dict.get("sma_50"),  # 使用 sma_50 作为 ma_99
        rsi=indicators_dict.get("rsi"),
        macd=macd,
        bollinger_bands=bollinger_bands,
        has_position=has_position,
        position_side=position.get("side") if has_position else None,
        position_size=position.get("size") if has_position else None,
        entry_price=position.get("entry_price") if has_position else None,
        unrealized_pnl=position.get("unrealized_pnl") if has_position else None,
        fear_greed_index=sentiment.get("fear_greed_index") if sentiment else None,
        market_sentiment=sentiment.get("sentiment") if sentiment else None,
        timestamp=datetime.now().isoformat(),
    )


def format_indicators_for_prompt(indicators_dict: Dict[str, Any]) -> str:
    """格式化技术指标为可读文本

    Args:
        indicators_dict: 技术指标字典

    Returns:
        格式化的文本
    """
    lines = []

    # 趋势指标
    if indicators_dict.get("sma_5"):
        lines.append(f"SMA5: {indicators_dict['sma_5']:.2f}")
    if indicators_dict.get("sma_20"):
        lines.append(f"SMA20: {indicators_dict['sma_20']:.2f}")
    if indicators_dict.get("sma_50"):
        lines.append(f"SMA50: {indicators_dict['sma_50']:.2f}")

    # 动量指标
    if indicators_dict.get("rsi"):
        rsi = indicators_dict["rsi"]
        rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
        lines.append(f"RSI(14): {rsi:.1f} ({rsi_status})")

    # ATR
    if indicators_dict.get("atr"):
        atr = indicators_dict["atr"]
        atr_pct = indicators_dict.get("atr_percent", 0)
        lines.append(f"ATR: {atr:.2f} ({atr_pct:.2f}%)")

    # ADX
    if indicators_dict.get("adx"):
        adx = indicators_dict["adx"]
        trend_strength = "强趋势" if adx > 25 else "弱趋势"
        lines.append(f"ADX: {adx:.1f} ({trend_strength})")

    # 市场状态
    if indicators_dict.get("market_regime"):
        lines.append(f"市场状态: {indicators_dict['market_regime']}")

    return "\n".join(lines) if lines else "暂无技术指标"
