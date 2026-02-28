"""
PerpBot AI 提示词模板模块

提供各种交易决策场景的提示词模板
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
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

    # 系统角色定义
    SYSTEM_ROLE = """你是一个专业的加密货币合约交易助手，具备以下能力：
1. 深刻理解加密货币市场动态和价格行为
2. 熟练运用技术分析指标（MA、RSI、MACD、布林带等）
3. 理解合约交易机制（杠杆、保证金、资金费率等）
4. 具备风险管理意识，注重资金安全
5. 能够综合多维度信息做出理性判断

你的决策原则：
- 风险控制优先，宁可错过也不做错
- 顺势而为，不逆势操作
- 严格执行止损止盈
- 保持冷静，不受情绪影响
- 承认不确定性，必要时选择观望

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
1. 综合考虑技术面、情绪面和持仓情况
2. 如果有持仓，优先考虑风险控制
3. 如果没有明确信号，选择HOLD
4. 给出具体的止损止盈价位

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
                bb_position = f"{((context.current_price - lower) / (upper - lower) * 100):.1f}%"

        return cls.TECHNICAL_INDICATORS_TEMPLATE.format(
            ma_7=context.ma_7 or "N/A",
            ma_25=context.ma_25 or "N/A",
            ma_99=context.ma_99 or "N/A",
            rsi=context.rsi or "N/A",
            macd_dif=macd_dif,
            macd_dea=macd_dea,
            macd_histogram=macd_hist,
            bb_upper=context.bollinger_bands.get("upper", "N/A") if context.bollinger_bands else "N/A",
            bb_middle=context.bollinger_bands.get("middle", "N/A") if context.bollinger_bands else "N/A",
            bb_lower=context.bollinger_bands.get("lower", "N/A") if context.bollinger_bands else "N/A",
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
            pnl_percent = (context.unrealized_pnl / (context.entry_price * context.position_size)) * 100

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
        prompt = f"""## 紧急平仓评估

### 当前持仓
- 方向: {'做多' if context.position_side == 'long' else '做空'}
- 入场价: {context.entry_price}
- 当前价: {context.current_price}
- 未实现盈亏: {context.unrealized_pnl:+.2f} USDT

### 市场状况
- 24小时涨跌: {context.price_change_24h:+.2f}%
- RSI: {context.rsi}
- 资金费率: {context.funding_rate}

### 评估要求
请评估是否需要立即平仓：
1. 如果亏损接近止损，建议CLOSE
2. 如果盈利达到目标，建议CLOSE
3. 如果市场出现反转信号，建议CLOSE
4. 否则建议HOLD

请返回JSON格式的决策结果："""
        return prompt

    @classmethod
    def build_risk_alert_prompt(cls, context: PromptContext, alert_type: str, alert_message: str) -> str:
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
- 方向: {'做多' if context.position_side == 'long' else '做空'}
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
    indicators: Dict[str, Any],
    position: Optional[Dict[str, Any]] = None,
    sentiment: Optional[Dict[str, Any]] = None,
) -> PromptContext:
    """创建提示词上下文的便捷函数

    Args:
        symbol: 交易对
        market_data: 市场数据
        indicators: 技术指标
        position: 持仓信息
        sentiment: 市场情绪

    Returns:
        PromptContext 实例
    """
    has_position = position is not None and position.get("size", 0) > 0

    return PromptContext(
        symbol=symbol,
        current_price=market_data.get("current_price", 0),
        price_change_24h=market_data.get("change_24h", 0),
        volume_24h=market_data.get("volume_24h", 0),
        high_24h=market_data.get("high_24h", 0),
        low_24h=market_data.get("low_24h", 0),
        funding_rate=market_data.get("funding_rate"),
        open_interest=market_data.get("open_interest"),
        ma_7=indicators.get("ma_7"),
        ma_25=indicators.get("ma_25"),
        ma_99=indicators.get("ma_99"),
        rsi=indicators.get("rsi"),
        macd=indicators.get("macd"),
        bollinger_bands=indicators.get("bollinger_bands"),
        has_position=has_position,
        position_side=position.get("side") if has_position else None,
        position_size=position.get("size") if has_position else None,
        entry_price=position.get("entry_price") if has_position else None,
        unrealized_pnl=position.get("unrealized_pnl") if has_position else None,
        fear_greed_index=sentiment.get("fear_greed_index") if sentiment else None,
        market_sentiment=sentiment.get("sentiment") if sentiment else None,
        timestamp=datetime.now().isoformat(),
    )