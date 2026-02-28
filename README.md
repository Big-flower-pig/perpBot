# PerpBot - AI 驱动的加密货币合约交易机器人

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个专业级的加密货币永续合约交易机器人，集成了 DeepSeek AI 进行智能决策分析。

## ✨ 特性

- 🤖 **AI 驱动决策** - 集成 DeepSeek API 进行智能交易分析
- 🧠 **AI 决策记忆** - 学习历史成功/失败模式，自适应优化决策
- 📊 **专业技术分析** - MA、RSI、MACD、布林带、ATR、ADX 等多种指标
- 📈 **市场状态识别** - 自动识别趋势/震荡/过渡市场状态
- ⚠️ **专业风险管理** - 凯利公式仓位、VaR风险计算、动态止损止盈
- 📉 **ATR 动态止损** - 基于波动率的智能止损，适应市场变化
- 🔄 **复利模式** - 支持复利交易，自动追踪本金变化
- 🔔 **实时通知** - Telegram 消息推送，随时掌握交易动态
- 📉 **异常检测** - 价格异常、成交量异常、波动率异常检测
- 🛠️ **模块化设计** - 清晰的代码架构，易于扩展和维护
- 📝 **完善日志** - 详细的日志记录，支持 JSON 格式输出
- 🐳 **Docker 支持** - 开箱即用的容器化部署方案

## 📋 目录

- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [Docker 部署](#docker-部署)
- [服务器部署](#服务器部署)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
- [风险控制](#风险控制)
- [开发指南](#开发指南)
- [免责声明](#免责声明)

## 🚀 安装

### 方式一：本地安装

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/perpBot.git
cd perpBot
```

#### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
```

### 方式二：Docker 安装（推荐）

确保已安装 Docker 和 Docker Compose：

```bash
# 检查 Docker 版本
docker --version
docker-compose --version
```

## ⚙️ 配置

### 环境变量 (.env)

```bash
# OKX 交易所 API
OKX_API_KEY=your_api_key
OKX_SECRET=your_secret
OKX_PASSWORD=your_password

# DeepSeek AI API
DEEPSEEK_API_KEY=your_deepseek_api_key

# Telegram 通知 (可选)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 配置文件 (config.yaml)

主要配置项：

```yaml
# 交易设置
trading:
  symbol: "BTC/USDT:USDT"  # 交易对
  leverage: 10             # 杠杆倍数
  interval: 60             # 交易周期（秒）
  margin_mode: "isolated"  # 保证金模式

# 仓位管理
position:
  mode: "compound"         # compound (复利) 或 fixed (固定)
  base_usdt_amount: 100    # 基础本金

# 风险控制
risk:
  max_position_size: 1000  # 最大持仓价值
  max_daily_loss: 50       # 日内最大亏损
  stop_loss_pct: 3.0       # 止损百分比
  take_profit_pct: 5.0     # 止盈百分比
```

## 📖 使用方法

### 本地运行

#### 模拟运行（推荐先测试）

```bash
python main.py --dry-run
```

#### 正式运行

```bash
python main.py
```

#### 指定配置文件

```bash
python main.py --config prod.yaml
```

#### 调整日志级别

```bash
python main.py --log-level DEBUG
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | config.yaml |
| `--dry-run` | 模拟运行模式（不下单） | False |
| `--log-level` | 日志级别 | INFO |
| `--version` | 显示版本号 | - |

## 🐳 Docker 部署

### 快速启动

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API 密钥

# 2. 构建并启动
docker-compose up -d

# 3. 查看日志
docker-compose logs -f perpbot
```

### Docker 常用命令

```bash
# 构建镜像
docker build -t perpbot:latest .

# 运行容器（模拟模式）
docker run -d \
  --name perpbot \
  --env-file .env \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  perpbot:latest \
  python main.py --dry-run

# 运行容器（正式模式）
docker run -d \
  --name perpbot \
  --env-file .env \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  --restart unless-stopped \
  perpbot:latest

# 查看容器日志
docker logs -f perpbot

# 进入容器
docker exec -it perpbot /bin/bash

# 停止容器
docker-compose down

# 重启容器
docker-compose restart
```

### Docker Compose 配置说明

```yaml
services:
  perpbot:
    build: .
    image: perpbot:latest
    container_name: perpbot
    environment:
      - TZ=Asia/Shanghai           # 时区设置
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - .env                        # 环境变量文件
    volumes:
      - ./config.yaml:/app/config.yaml:ro  # 配置文件挂载
      - perpbot_data:/app/data      # 数据持久化
    restart: unless-stopped         # 自动重启
    deploy:
      resources:
        limits:
          cpus: '1.0'               # CPU 限制
          memory: 512M              # 内存限制
```

## 🖥️ 服务器部署

### 部署前准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 Docker Compose
sudo apt install docker-compose -y

# 验证安装
docker --version
docker-compose --version
```

### 部署步骤

#### 1. 上传项目到服务器

```bash
# 方式一：Git 克隆
git clone https://github.com/yourusername/perpBot.git
cd perpBot

# 方式二：SCP 上传
scp -r perpBot user@server:/home/user/
```

#### 2. 配置环境

```bash
# 创建环境变量文件
cp .env.example .env
nano .env  # 填入 API 密钥

# 根据需要修改配置
nano config.yaml
```

#### 3. 启动服务

```bash
# 构建并后台启动
docker-compose up -d --build

# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 使用 systemd 管理（可选）

创建 systemd 服务文件：

```bash
sudo nano /etc/systemd/system/perpbot.service
```

内容：

```ini
[Unit]
Description=PerpBot Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/perpBot
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable perpbot
sudo systemctl start perpbot
sudo systemctl status perpbot
```

### 设置开机自启

```bash
# Docker 服务开机自启
sudo systemctl enable docker

# PerpBot 容器自动重启（已在 docker-compose.yml 中配置）
# restart: unless-stopped
```

### 日志管理

```bash
# 查看实时日志
docker-compose logs -f --tail=100

# 导出日志
docker-compose logs > perpbot.log

# 日志文件位置
ls -la data/logs/
```

### 监控与告警

```bash
# 查看容器资源使用
docker stats perpbot

# 查看容器健康状态
docker inspect --format='{{.State.Health.Status}}' perpbot
```

### 更新部署

```bash
# 拉取最新代码
git pull origin main

# 重新构建并启动
docker-compose down
docker-compose up -d --build

# 或者一步完成
docker-compose up -d --build
```

### 备份数据

```bash
# 备份数据目录
tar -czvf perpbot_backup_$(date +%Y%m%d).tar.gz data/

# 备份配置
tar -czvf perpbot_config_$(date +%Y%m%d).tar.gz config.yaml .env
```

## 📁 项目结构

```
perpBot/
├── main.py                 # 主程序入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
├── .env.example           # 环境变量示例
├── .gitignore             # Git 忽略配置
├── README.md              # 项目文档
├── ARCHITECTURE.md        # 架构文档
│
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── logger.py          # 日志系统
│   ├── config.py          # 配置管理
│   ├── decorators.py      # 装饰器
│   ├── helpers.py         # 辅助函数
│   └── validators.py      # 参数验证
│
├── core/                   # 核心模块
│   ├── __init__.py
│   ├── exchange.py        # 交易所管理
│   ├── strategy.py        # 策略引擎
│   ├── risk_manager.py    # 风险管理
│   └── position.py        # 仓位管理
│
├── ai/                     # AI 模块
│   ├── __init__.py
│   ├── deepseek_analyzer.py  # DeepSeek 分析器
│   └── prompt_templates.py   # 提示词模板
│
├── monitoring/             # 监控模块
│   ├── __init__.py
│   ├── price_monitor.py      # 价格监控
│   └── anomaly_detector.py   # 异常检测
│
├── notification/           # 通知模块
│   ├── __init__.py
│   └── telegram.py        # Telegram 通知
│
└── data/                   # 数据目录
    ├── logs/              # 日志文件
    └── tracking/          # 追踪数据
```

## 🔧 核心模块

### ExchangeManager (交易所管理)

```python
from core import get_exchange_manager

exchange = get_exchange_manager()
exchange.connect()

# 获取行情
ticker = exchange.get_ticker("BTC/USDT:USDT")

# 获取 K 线
ohlcv = exchange.get_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=100)

# 下单
result = exchange.create_market_order("BTC/USDT:USDT", "buy", 0.1)
```

### StrategyEngine (策略引擎) - 专业版

```python
from core import StrategyEngine

strategy = StrategyEngine()

# 计算技术指标（包含 ATR、ADX）
indicators = strategy.calculate_indicators(ohlcv)

# 市场状态识别
# TRENDING: 趋势市场，ADX > 25
# RANGING: 震荡市场，ADX < 20
# TRANSITIONAL: 过渡状态
market_regime = indicators.market_regime

# ATR 动态止损
stop_loss, take_profit = strategy._calculate_dynamic_stops(
    current_price=95000,
    atr=500,
    side="long"
)

# 根据市场状态使用不同策略
if market_regime == "TRENDING":
    signal = strategy._trend_strategy(indicators)
elif market_regime == "RANGING":
    signal = strategy._range_strategy(indicators)
```

### RiskManager (风险管理) - 专业版

```python
from core import get_risk_manager

risk = get_risk_manager()

# 凯利公式仓位计算
position_size = risk.calculate_position_size(
    capital=1000,
    price=95000,
    confidence="HIGH",
    use_kelly=True  # 启用凯利公式
)

# VaR 风险计算
var_result = risk.calculate_var(
    capital=1000,
    confidence_level=0.95,
    holding_period_days=1
)
print(f"95%置信度下最大可能损失: {var_result['var']:.2f} USDT")

# 获取性能指标
metrics = risk.get_performance_metrics()
print(f"胜率: {metrics.win_rate:.1f}%")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"凯利比例: {metrics.kelly_fraction:.1%}")

# 评估交易
assessment = risk.assess_trade(
    symbol="BTC/USDT:USDT",
    side="long",
    confidence="HIGH",
    current_price=95000,
)

if assessment.approved:
    print("交易通过风险评估")
```

### DeepSeekAnalyzer (AI 分析) - 增强版

```python
from ai import get_analyzer

analyzer = get_analyzer()

# 分析市场（自动注入历史经验）
decision = analyzer.analyze(
    market_data={...},
    indicators={...},
    position={...},
)

print(f"决策: {decision.decision.value}")
print(f"信心: {decision.confidence.value}")
print(f"市场状态: {decision.market_regime}")
print(f"理由: {decision.reason}")

# 记录决策结果（用于学习）
analyzer.record_outcome(
    decision_id="20240101_123456_BUY",
    pnl=15.50,
    exit_price=95500,
    holding_time_minutes=30
)

# 获取记忆系统统计
stats = analyzer.get_memory_stats()
print(f"总体胜率: {stats['memory']['overall_win_rate']:.1f}%")
print(f"最佳市场状态: {stats['memory']['best_regime']}")
```

## 🛡️ 风险控制

### 多层风险控制机制

1. **交易前评估**
   - 日内亏损限制
   - 持仓数量限制
   - 连续亏损限制

2. **仓位管理 (凯利公式)**
   - 基于历史胜率和盈亏比计算最优仓位
   - 半凯利策略降低风险
   - 信心程度调整
   - 最大仓位限制 (50%)

3. **止损止盈 (ATR 动态)**
   - 基于 ATR 自动计算止损止盈价位
   - 趋势市场: 2倍ATR止损, 3倍ATR止盈
   - 震荡市场: 1.5倍ATR止损, 2倍ATR止盈

4. **VaR 风险管理**
   - 95%置信度下最大可能损失计算
   - 历史模拟法
   - 持有期调整

5. **实时监控**
   - 持仓风险评估
   - 价格异常检测
   - 自动平仓机制

### 风险等级

| 等级 | 描述 | 行为 |
|------|------|------|
| LOW | 低风险 | 正常交易 |
| MEDIUM | 中风险 | 减少仓位 |
| HIGH | 高风险 | 禁止开仓 |
| CRITICAL | 极高风险 | 强制平仓 |

### 市场状态识别

| 状态 | ADX 值 | 特征 | 策略 |
|------|--------|------|------|
| TRENDING | > 25 | 强趋势 | 趋势跟踪 |
| RANGING | < 20 | 震荡区间 | 支撑阻力 |
| TRANSITIONAL | 20-25 | 过渡状态 | 谨慎观望 |

## 👨‍💻 开发指南

### 添加新策略

1. 在 `core/strategy.py` 中添加新指标计算方法
2. 在 `_calculate_indicators()` 中集成新指标
3. 更新 `TechnicalIndicators` 数据类
4. 在对应市场状态策略中集成

### 扩展 AI 记忆系统

```python
# 添加自定义记忆模式
analyzer._memory.success_patterns.append({
    "decision": "BUY",
    "market_regime": "TRENDING",
    "key_factors": ["RSI超卖", "MACD金叉"],
})

# 获取经验教训
lessons = analyzer._memory.get_lessons_learned()
```

### 添加新通知渠道

1. 在 `notification/` 目录创建新模块
2. 实现 `send()` 方法
3. 在 `config.yaml` 中添加配置

### 代码风格

- 使用 Python 3.9+ 特性
- 遵循 PEP 8 规范
- 使用类型注解
- 编写文档字符串

### 运行测试

```bash
# 安装开发依赖
pip install -r requirements.txt

# 验证模块导入
python -c "from core.risk_manager import RiskManager; print('OK')"

# 运行测试
python -m pytest tests/ -v
```

## ❓ 常见问题

### Q: 如何修改交易对？

A: 编辑 `config.yaml` 中的 `trading.symbol` 配置项。

### Q: 如何查看交易日志？

A:
```bash
# Docker 部署
docker-compose logs -f perpbot

# 本地运行
tail -f data/logs/perpbot.log
```

### Q: 如何切换到模拟模式？

A:
```bash
# 命令行参数
python main.py --dry-run

# 或修改 config.yaml
trading:
  dry_run: true
```

### Q: 容器无法启动怎么办？

A:
```bash
# 检查日志
docker-compose logs perpbot

# 检查配置
docker-compose config

# 重新构建
docker-compose down
docker-compose up -d --build
```

## ⚠️ 免责声明

**本软件仅供学习和研究目的使用。**

- 加密货币交易存在极高风险，可能导致本金全部损失
- 本机器人不保证盈利，过往表现不代表未来收益
- 使用本软件进行实盘交易的所有风险由用户自行承担
- 开发者不对任何因使用本软件造成的损失负责

**使用前请务必：**
1. 充分了解加密货币市场风险
2. 在模拟环境充分测试
3. 仅使用可承受损失的资金
4. 设置合理的止损策略

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📞 联系方式

- 作者: Your Name
- Email: your.email@example.com
- Telegram: @your_username

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**