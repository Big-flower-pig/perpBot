# PerpBot 量化交易系统架构文档

## 📋 项目概述

PerpBot 是一个专业的加密货币永续合约量化交易系统，采用模块化架构设计，支持 AI 驱动的交易策略、实时风险监控、回测框架和多交易所接入。

---

## 🏗️ 系统架构

```
perpBot/
├── main.py                     # 主程序入口
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖包
├── README.md                   # 项目文档
├── ARCHITECTURE.md             # 架构文档（本文件）
│
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── exchange.py             # 交易所接口封装
│   ├── strategy.py             # 交易策略引擎
│   ├── risk_manager.py         # 风险管理模块
│   ├── position.py             # 仓位管理
│   └── order.py                # 订单管理
│
├── ai/                         # AI 决策模块
│   ├── __init__.py
│   ├── deepseek_analyzer.py    # DeepSeek 分析器
│   ├── prompt_templates.py     # Prompt 模板
│   └── decision_logger.py      # 决策日志记录
│
├── monitoring/                 # 监控模块
│   ├── __init__.py
│   ├── price_monitor.py        # 价格监控
│   ├── anomaly_detector.py     # 异常检测
│   ├── alert_manager.py        # 告警管理
│   └── metrics.py              # Prometheus 指标
│
├── backtest/                   # 回测框架
│   ├── __init__.py
│   ├── engine.py               # 回测引擎
│   ├── data_loader.py          # 数据加载
│   ├── performance.py          # 性能评估
│   └── report.py               # 报告生成
│
├── database/                   # 数据库模块
│   ├── __init__.py
│   ├── models.py               # 数据模型
│   ├── repository.py           # 数据仓库
│   └── migrations/             # 数据库迁移
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── logger.py               # 日志系统
│   ├── config.py               # 配置管理
│   ├── helpers.py              # 辅助函数
│   ├── decorators.py           # 装饰器
│   └── validators.py           # 验证器
│
├── notification/               # 通知模块
│   ├── __init__.py
│   ├── telegram.py             # Telegram 通知
│   └── base.py                 # 通知基类
│
├── tests/                      # 测试模块
│   ├── __init__.py
│   ├── conftest.py             # pytest 配置
│   ├── test_exchange.py
│   ├── test_strategy.py
│   ├── test_risk_manager.py
│   └── test_backtest.py
│
├── data/                       # 数据目录
│   ├── historical/             # 历史数据
│   ├── cache/                  # 缓存数据
│   └── tracking/               # 追踪数据
│
└── logs/                       # 日志目录
    ├── trading.log             # 交易日志
    ├── error.log               # 错误日志
    └── monitor.log             # 监控日志
```

---

## 🔧 核心模块设计

### 1. Exchange 模块 (`core/exchange.py`)

**职责**：封装交易所 API，提供统一的交易接口

```python
class ExchangeManager:
    """交易所管理器"""

    - 初始化交易所连接
    - 验证 API 连接
    - 设置杠杆和仓位模式
    - 获取市场数据（K线、实时价格）
    - 执行订单（开仓、平仓、止损止盈）
    - 获取账户信息（余额、持仓）
    - 自动重连机制
    - API 限流处理
```

**关键特性**：
- 支持多交易所（OKX、Binance 等）
- 自动重连和错误恢复
- 请求限流保护
- 沙盒模式支持

### 2. Strategy 模块 (`core/strategy.py`)

**职责**：实现交易策略逻辑

```python
class StrategyEngine:
    """策略引擎"""

    - 技术指标计算（MA、RSI、MACD、布林带等）
    - 趋势分析
    - 支撑阻力位识别
    - 信号生成
    - 策略参数优化
```

```python
class AIStrategy:
    """AI 驱动的策略"""

    - 调用 DeepSeek API 分析市场
    - 解析 AI 决策
    - 决策置信度评估
    - 历史决策准确率追踪
```

### 3. Risk Manager 模块 (`core/risk_manager.py`)

**职责**：风险管理和资金控制

```python
class RiskManager:
    """风险管理器"""

    - 止损止盈计算
    - 仓位大小计算（凯利公式、风险平价）
    - 最大回撤控制
    - VaR 计算
    - 交易频率限制
    - 日内亏损限制
    - 波动率自适应止损
```

**风控规则**：
1. 单笔交易风险不超过本金 X%
2. 总持仓不超过本金 Y%
3. 日内最大亏损 Z%
4. 连续亏损后降低仓位
5. 高波动期间降低仓位

### 4. Position 模块 (`core/position.py`)

**职责**：仓位管理

```python
class PositionManager:
    """仓位管理器"""

    - 获取当前持仓
    - 计算仓位价值
    - 计算盈亏
    - 复利模式管理
    - 本金追踪
```

---

## 🤖 AI 决策模块设计

### DeepSeek Analyzer (`ai/deepseek_analyzer.py`)

```python
class DeepSeekAnalyzer:
    """DeepSeek 分析器"""

    - 构建分析 Prompt
    - 调用 DeepSeek API
    - 解析返回结果
    - 决策日志记录
    - 响应时间监控
    - 异常处理和重试
```

### Prompt 模板 (`ai/prompt_templates.py`)

```python
class PromptTemplates:
    """Prompt 模板管理"""

    - 市场分析模板
    - 异常决策模板
    - 风险评估模板
    - 多时间框架分析模板
```

### 决策日志 (`ai/decision_logger.py`)

```python
class DecisionLogger:
    """决策日志记录器"""

    - 记录每次 AI 决策
    - 追踪决策准确率
    - 分析决策模式
    - 生成决策报告
```

---

## 📡 监控模块设计

### Price Monitor (`monitoring/price_monitor.py`)

```python
class PriceMonitor:
    """价格监控器"""

    - 实时价格获取
    - 价格变化追踪
    - 线程安全设计
    - 异步数据获取
```

### Anomaly Detector (`monitoring/anomaly_detector.py`)

```python
class AnomalyDetector:
    """异常检测器"""

    - 价格急变检测
    - 成交量异常检测
    - 突破关键位检测
    - 持仓风险检测
    - 市场情绪突变检测
```

### Alert Manager (`monitoring/alert_manager.py`)

```python
class AlertManager:
    """告警管理器"""

    - 告警级别定义（INFO、WARNING、CRITICAL）
    - 告警去重
    - 告警冷却
    - 多渠道通知
```

### Metrics (`monitoring/metrics.py`)

```python
class Metrics:
    """Prometheus 指标"""

    - 交易次数计数
    - 盈亏统计
    - API 响应时间
    - 系统健康状态
```

---

## 📊 回测框架设计

### Backtest Engine (`backtest/engine.py`)

```python
class BacktestEngine:
    """回测引擎"""

    - 加载历史数据
    - 模拟交易执行
    - 计算性能指标
    - 生成回测报告
```

### Performance (`backtest/performance.py`)

```python
class PerformanceAnalyzer:
    """性能分析器"""

    - 总收益率
    - 年化收益率
    - 最大回撤
    - Sharpe Ratio
    - Sortino Ratio
    - 胜率
    - 盈亏比
    - Calmar Ratio
```

---

## 💾 数据库设计

### 数据模型 (`database/models.py`)

```python
# 使用 SQLAlchemy 或 Peewee

class Trade(BaseModel):
    """交易记录"""
    id: int
    symbol: str
    side: str  # long/short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    timestamp: datetime
    strategy: str
    confidence: str

class Signal(BaseModel):
    """信号记录"""
    id: int
    symbol: str
    signal: str  # BUY/SELL/HOLD
    reason: str
    confidence: str
    timestamp: datetime
    ai_response: str

class CapitalTracking(BaseModel):
    """本金追踪"""
    id: int
    initial_capital: float
    current_capital: float
    total_pnl: float
    last_update: datetime
```

---

## 🔧 工具模块设计

### Logger (`utils/logger.py`)

```python
class TradingLogger:
    """交易日志系统"""

    - 多级别日志（DEBUG、INFO、WARNING、ERROR、CRITICAL）
    - 日志文件轮转
    - 彩色终端输出
    - 结构化日志（JSON）
    - 敏感信息脱敏
```

### Config (`utils/config.py`)

```python
class ConfigManager:
    """配置管理器"""

    - YAML 配置加载
    - 配置验证（schema）
    - 敏感信息加密
    - 热重载支持
    - 环境变量覆盖
```

### Decorators (`utils/decorators.py`)

```python
# 常用装饰器

@retry_on_failure(max_retries=3, delay=1)
@rate_limit(calls_per_second=10)
@log_execution_time
@thread_safe
@validate_params
```

---

## 🔔 通知模块设计

### Telegram (`notification/telegram.py`)

```python
class TelegramNotifier:
    """Telegram 通知器"""

    - 发送交易通知
    - 发送告警通知
    - 发送每日报告
    - 支持富文本格式
    - 异步发送
```

---

## 🔄 执行流程

### 主流程

```
1. 启动初始化
   ├── 加载配置
   ├── 初始化日志
   ├── 连接交易所
   ├── 设置杠杆和仓位模式
   └── 发送启动通知

2. 定时任务循环
   ├── 获取市场数据
   ├── 计算技术指标
   ├── AI 分析决策
   ├── 风险评估
   ├── 执行交易
   └── 记录日志

3. 实时监控线程
   ├── 价格监控
   ├── 异常检测
   ├── 风险监控
   └── 触发告警

4. 关闭流程
   ├── 停止监控线程
   ├── 保存状态
   ├── 发送关闭通知
   └── 关闭日志
```

---

## 🛡️ 安全设计

1. **API Key 保护**
   - 环境变量存储
   - 加密存储可选
   - 不记录到日志

2. **交易安全**
   - 订单确认机制
   - 异常订单检测
   - 交易限额

3. **数据安全**
   - 敏感数据加密
   - 数据库备份
   - 访问控制

---

## 📈 性能优化

1. **数据缓存**
   - K 线数据缓存
   - 技术指标缓存
   - Redis 可选

2. **并发处理**
   - 异步 I/O（asyncio）
   - 线程池
   - 进程池（CPU 密集任务）

3. **资源管理**
   - 连接池
   - 内存限制
   - 定期清理

---

## 📦 依赖包

```txt
# 核心
ccxt>=4.0.0
pandas>=2.0.0
numpy>=1.24.0

# AI
openai>=1.0.0

# 数据库
sqlalchemy>=2.0.0
aiosqlite>=0.19.0

# Web（可选）
fastapi>=0.100.0
uvicorn>=0.23.0

# 监控
prometheus-client>=0.17.0

# 通知
requests>=2.31.0

# 测试
pytest>=7.4.0
pytest-asyncio>=0.21.0

# 工具
pyyaml>=6.0
python-dotenv>=1.0.0
cryptography>=41.0.0
schedule>=1.2.0
```

---

## 🚀 实施计划

### Phase 1: 基础框架（第1-2天）
- [x] 创建项目结构
- [ ] 实现配置管理
- [ ] 实现日志系统
- [ ] 实现交易所接口

### Phase 2: 核心功能（第3-4天）
- [ ] 实现策略引擎
- [ ] 实现 AI 分析器
- [ ] 实现仓位管理
- [ ] 实现风险管理

### Phase 3: 监控系统（第5天）
- [ ] 实现价格监控
- [ ] 实现异常检测
- [ ] 实现告警系统

### Phase 4: 回测框架（第6天）
- [ ] 实现回测引擎
- [ ] 实现性能分析
- [ ] 实现报告生成

### Phase 5: 数据库和测试（第7天）
- [ ] 实现数据库模块
- [ ] 编写单元测试
- [ ] 集成测试

---

## 📝 配置文件示例 (config.yaml)

```yaml
# 交易所配置
exchange:
  name: okx
  sandbox: true
  api_key: ${OKX_API_KEY}
  secret: ${OKX_SECRET}
  password: ${OKX_PASSWORD}

# 交易配置
trading:
  symbol: BTC/USDT:USDT
  leverage: 10
  timeframe: 15m
  margin_mode: isolated  # cross / isolated

# AI 配置
ai:
  provider: deepseek
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-chat
  temperature: 0.1

# 仓位管理
position:
  mode: compound  # fixed / compound
  initial_capital: 100
  max_position_ratio: 1.0
  confidence_multipliers:
    high: 1.0
    medium: 0.8
    low: 0.5

# 风险管理
risk:
  stop_loss_percent: 3.0
  take_profit_percent: 5.0
  max_daily_loss: 10.0
  max_trades_per_day: 20
  trailing_stop_percent: 2.0

# 监控配置
monitoring:
  enabled: true
  check_interval: 10
  price_change_threshold: 1.0
  rapid_change_threshold: 2.0

# 通知配置
notification:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}

# 日志配置
logging:
  level: INFO
  file: logs/trading.log
  max_size: 10MB
  backup_count: 5

# 数据库配置
database:
  type: sqlite
  path: data/perpbot.db
```

---

## ✅ 验收标准

1. **功能完整性**
   - 所有模块正常工作
   - 交易执行正确
   - 监控告警正常

2. **性能指标**
   - API 响应时间 < 500ms
   - 内存占用 < 500MB
   - CPU 占用 < 10%（空闲时）

3. **稳定性**
   - 7x24 小时无故障运行
   - 异常自动恢复
   - 数据不丢失

4. **可维护性**
   - 代码覆盖率 > 80%
   - 文档完整
   - 日志完善

---

*文档版本: 1.0*
*最后更新: 2026-02-28*