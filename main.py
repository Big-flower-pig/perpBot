#!/usr/bin/env python3
"""
PerpBot - AI 驱动的加密货币合约交易机器人

主程序入口，协调各模块运行

Usage:
    python main.py                    # 正常运行
    python main.py --config prod.yaml # 指定配置文件
    python main.py --dry-run          # 模拟运行（不下单）
    python main.py --backtest         # 回测模式
"""

import argparse
import signal
import sys
import time
import threading
from datetime import datetime
from typing import Optional

from utils.logger import get_logger, setup_logging
from utils.config import get_config, load_config
from utils.validators import validate_config
from core.exchange import get_exchange_manager
from core.strategy import StrategyEngine
from core.risk_manager import get_risk_manager
from core.position import get_position_manager
from ai.deepseek_analyzer import get_analyzer, Decision
from monitoring.price_monitor import get_price_monitor
from monitoring.anomaly_detector import get_anomaly_detector
from notification.telegram import get_telegram_notifier


class TradingBot:
    """交易机器人主类

    协调各模块运行，执行交易逻辑

    Example:
        bot = TradingBot()
        bot.start()
    """

    VERSION = "2.0.0"

    def __init__(self, dry_run: bool = False):
        """初始化交易机器人

        Args:
            dry_run: 模拟运行模式（不下单）
        """
        self._logger = get_logger("bot")
        self._dry_run = dry_run

        # 运行状态
        self._running = False
        self._stop_event = threading.Event()

        # 模块实例
        self._exchange = None
        self._strategy = None
        self._risk_manager = None
        self._position_manager = None
        self._analyzer = None
        self._price_monitor = None
        self._anomaly_detector = None
        self._notifier = None

        # 统计
        self._start_time: Optional[datetime] = None
        self._loop_count = 0
        self._trade_count = 0

        # 交易冷却控制
        self._last_trade_time: Optional[datetime] = None  # 最后一次交易时间
        self._last_close_time: Optional[datetime] = None  # 最后一次平仓时间
        self._last_trade_side: Optional[str] = None  # 最后一次交易方向

    def initialize(self) -> bool:
        """初始化所有模块

        Returns:
            是否初始化成功
        """
        try:
            self._logger.info("=" * 50)
            self._logger.info(f"PerpBot v{self.VERSION} 初始化中...")
            self._logger.info("=" * 50)

            # 验证配置
            self._logger.info("验证配置...")
            validate_config()
            self._logger.info("配置验证通过")

            # 初始化交易所管理器
            self._logger.info("初始化交易所连接...")
            self._exchange = get_exchange_manager()
            # dry-run 模式也需要连接交易所获取市场数据，只是不执行实际交易
            if not self._exchange.connect():
                self._logger.error("交易所连接失败")
                return False
            self._logger.info("交易所连接成功")

            # 初始化策略引擎
            self._logger.info("初始化策略引擎...")
            self._strategy = StrategyEngine()

            # 初始化风险管理器
            self._logger.info("初始化风险管理器...")
            self._risk_manager = get_risk_manager()

            # 初始化仓位管理器
            self._logger.info("初始化仓位管理器...")
            self._position_manager = get_position_manager()

            # 初始化 AI 分析器
            self._logger.info("初始化 AI 分析器...")
            self._analyzer = get_analyzer()

            # 初始化价格监控器
            self._logger.info("初始化价格监控器...")
            self._price_monitor = get_price_monitor()

            # 初始化异常检测器
            self._logger.info("初始化异常检测器...")
            self._anomaly_detector = get_anomaly_detector()

            # 初始化通知器
            self._logger.info("初始化通知器...")
            self._notifier = get_telegram_notifier()

            self._logger.info("=" * 50)
            self._logger.info("所有模块初始化完成!")
            self._logger.info("=" * 50)

            return True

        except Exception as e:
            self._logger.error(f"初始化失败: {e}", exc_info=True)
            return False

    def start(self):
        """启动交易机器人"""
        if self._running:
            self._logger.warning("机器人已在运行中")
            return

        # 初始化
        if not self.initialize():
            self._logger.error("初始化失败，无法启动")
            return

        self._running = True
        self._start_time = datetime.now()

        # 发送启动通知
        self._notifier.send_startup(self.VERSION)

        # 注册信号处理
        self._register_signals()

        self._logger.info(
            f"交易机器人启动 - 模式: {'模拟' if self._dry_run else '实盘'}"
        )

        try:
            # 主循环
            self._main_loop()
        except KeyboardInterrupt:
            self._logger.info("收到键盘中断信号")
        except Exception as e:
            self._logger.error(f"运行时错误: {e}", exc_info=True)
            self._notifier.send_error("运行时错误", str(e))
        finally:
            self.stop()

    def stop(self):
        """停止交易机器人"""
        if not self._running:
            return

        self._logger.info("正在停止交易机器人...")
        self._running = False
        self._stop_event.set()

        # 退出时检查持仓
        self._check_position_on_shutdown()

        # 断开交易所连接
        if self._exchange:
            try:
                self._exchange.disconnect()
            except Exception:
                pass

        # 发送停止通知
        try:
            runtime = datetime.now() - self._start_time if self._start_time else None
            runtime_str = str(runtime).split(".")[0] if runtime else "N/A"
            self._notifier.send_shutdown(
                f"运行时间: {runtime_str}, 交易次数: {self._trade_count}"
            )
        except Exception:
            pass

        self._logger.info("交易机器人已停止")

    def _check_position_on_shutdown(self):
        """退出时检查持仓，让 AI 分析并给出建议"""
        try:
            # 检查是否启用退出持仓检查
            shutdown_config = get_config("advanced.shutdown", {})
            if not shutdown_config.get("check_position", True):
                return

            self._logger.info("检查当前持仓状态...")

            # 获取所有持仓
            position = self._exchange.get_position()

            if not position:
                self._logger.info("当前无持仓，可以安全退出")
                return

            # 计算盈亏百分比
            pnl = position.unrealized_pnl or 0
            leverage = position.leverage or 10  # 默认杠杆为 10
            entry_price = position.entry_price or 0
            size = position.size or 0

            # 安全计算入场价值
            if entry_price > 0 and leverage > 0:
                entry_value = size * entry_price / leverage
            else:
                entry_value = 0

            pnl_percent = (pnl / entry_value * 100) if entry_value > 0 else 0

            # 显示持仓信息
            self._logger.info("=" * 50)
            self._logger.info("当前持仓状态:")
            self._logger.info(f"  交易对: {position.symbol}")
            self._logger.info(f"  方向: {'多头' if position.is_long else '空头'}")
            self._logger.info(f"  数量: {position.size}")
            entry_price_val = position.entry_price or 0
            self._logger.info(f"  入场价: {entry_price_val:.4f}")
            self._logger.info(f"  未实现盈亏: {pnl:+.2f} USDT ({pnl_percent:+.2f}%)")
            leverage_val = position.leverage or 10
            self._logger.info(f"  杠杆: {leverage_val}x")
            self._logger.info("=" * 50)

            # 获取当前价格和技术指标
            try:
                ticker = self._exchange.get_ticker(position.symbol)
                current_price = ticker.last if ticker else position.entry_price

                ohlcv = self._exchange.get_ohlcv(
                    position.symbol, timeframe="1h", limit=100
                )
                indicators = self._strategy.calculate_indicators(ohlcv) if ohlcv else {}
            except Exception as e:
                self._logger.warning(f"获取市场数据失败，使用默认值: {e}")
                current_price = position.entry_price
                indicators = {}

            # 准备市场数据
            market_data = {
                "current_price": current_price,
                "change_24h": 0,
                "volume_24h": 0,
                "high_24h": current_price,
                "low_24h": current_price,
            }

            # 准备持仓数据
            position_data = {
                "side": position.side,
                "size": position.size,
                "entry_price": position.entry_price,
                "unrealized_pnl": position.unrealized_pnl,
                "pnl_percent": pnl_percent,
            }

            # 调用 AI 分析持仓
            self._logger.info("AI 正在分析持仓，请稍候...")
            ai_decision = self._analyzer.analyze(
                market_data=market_data,
                indicators=indicators,
                position=position_data,
            )

            # 显示 AI 建议
            self._logger.info("=" * 50)
            self._logger.info("AI 持仓分析建议:")
            self._logger.info(f"  决策: {ai_decision.decision.value}")
            self._logger.info(f"  信心: {ai_decision.confidence.value}")
            self._logger.info(f"  理由: {ai_decision.reason}")
            stop_loss_val = ai_decision.stop_loss or 0
            take_profit_val = ai_decision.take_profit or 0
            if stop_loss_val > 0:
                self._logger.info(f"  建议止损: {stop_loss_val:.4f}")
            if take_profit_val > 0:
                self._logger.info(f"  建议止盈: {take_profit_val:.4f}")
            self._logger.info("=" * 50)

            # 发送通知
            pnl_status = "盈利" if pnl > 0 else "亏损" if pnl < 0 else "持平"
            action_suggestion = (
                "建议平仓" if ai_decision.decision == Decision.CLOSE else "建议继续持仓"
            )

            self._notifier.send_alert(
                title=f"退出分析 - {pnl_status} | {action_suggestion}",
                content=f"持仓: {position.symbol} {position.side} {position.size}张\n"
                f"盈亏: {pnl:+.2f} USDT ({pnl_percent:+.2f}%)\n\n"
                f"AI 建议: {ai_decision.decision.value}\n"
                f"理由: {ai_decision.reason}\n\n"
                f"止损: {stop_loss_val:.4f if stop_loss_val > 0 else '未设置'}\n"
                f"止盈: {take_profit_val:.4f if take_profit_val > 0 else '未设置'}",
                severity="HIGH" if ai_decision.decision == Decision.CLOSE else "MEDIUM",
            )

            # 根据 AI 决策处理
            if ai_decision.decision == Decision.CLOSE:
                self._logger.warning("AI 建议平仓！")
                # 在非 dry-run 模式下可以选择自动平仓
                auto_close = shutdown_config.get("auto_close_on_ai_advice", False)
                if auto_close and not self._dry_run:
                    self._logger.info("正在执行 AI 建议平仓...")
                    result = self._exchange.close_position(position=position)
                    if result.success:
                        self._logger.info(f"平仓成功: 订单ID {result.order_id}")
                    else:
                        self._logger.error(f"平仓失败: {result.error}")
                else:
                    self._logger.info("请手动决定是否平仓")
            else:
                self._logger.info("AI 建议继续持仓，退出时保持当前仓位")
                # 可以选择设置止损止盈订单（如果交易所支持）
                if stop_loss_val > 0 or take_profit_val > 0:
                    self._logger.info("建议手动设置止损止盈订单以控制风险")

        except Exception as e:
            self._logger.warning(f"检查持仓状态失败: {e}")

    def _register_signals(self):
        """注册信号处理函数"""
        signal.signal(signal.SIGINT, self._signal_handler)
        # SIGTERM 在 Windows 平台不存在
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        self._logger.info(f"收到信号 {signum}")
        self.stop()

    def _main_loop(self):
        """主循环"""
        interval = get_config("trading.interval", 60)
        check_interval = 0.5  # 每 0.5 秒检查一次停止信号

        while self._running:
            self._loop_count += 1

            try:
                # 执行一次交易循环
                self._trading_cycle()

            except Exception as e:
                self._logger.error(f"交易循环错误: {e}", exc_info=True)

            # 等待下一个周期（使用短间隔轮询，便于快速响应退出信号）
            wait_time = 0
            while wait_time < interval and self._running:
                self._stop_event.wait(check_interval)
                wait_time += check_interval
                if self._stop_event.is_set():
                    break

    def _trading_cycle(self):
        """单次交易循环"""
        symbol = get_config("trading.symbol")
        self._logger.info(
            f"--- 交易循环 #{self._loop_count} @ {datetime.now().strftime('%H:%M:%S')} ---"
        )

        # 1. 获取市场数据
        self._logger.debug("获取市场数据...")
        ticker = self._exchange.get_ticker(symbol)
        if not ticker:
            self._logger.warning("无法获取行情数据，跳过本次循环")
            return

        current_price = ticker.last
        self._logger.info(f"当前价格: {current_price}")

        # 更新价格监控器
        alerts = self._price_monitor.update(
            price=current_price,
            volume=ticker.quote_volume,
        )

        # 处理价格警报
        for alert in alerts:
            self._logger.warning(f"价格警报: {alert.message}")
            self._notifier.send_alert(
                title="价格警报",
                content=alert.message,
                severity="MEDIUM",
            )

        # 2. 获取 K 线数据并计算技术指标
        self._logger.debug("计算技术指标...")
        ohlcv = self._exchange.get_ohlcv(symbol, timeframe="1h", limit=200)
        if ohlcv:
            # calculate_indicators 和 analyze_trend 都需要 df 和 indicators
            indicators = self._strategy.calculate_indicators(ohlcv)
            trend = self._strategy.analyze_trend(ohlcv, indicators)

            self._logger.info(
                f"趋势分析: {trend.overall.value} (强度: {trend.strength:.1f})"
            )

            # 更新异常检测器
            self._anomaly_detector.add_price(current_price)
            anomalies = self._anomaly_detector.detect()

            for anomaly in anomalies:
                self._logger.warning(f"检测到异常: {anomaly.description}")
                if anomaly.severity.value in ["HIGH", "CRITICAL"]:
                    self._notifier.send_alert(
                        title="市场异常",
                        content=anomaly.description,
                        severity=anomaly.severity.value,
                    )
        else:
            indicators = {}
            trend = None

        # 3. 获取当前持仓
        self._logger.debug("检查持仓...")
        position = None
        try:
            position = self._exchange.get_position(symbol)
        except Exception as e:
            # dry-run 模式下，模拟盘 API Key 可能无法获取持仓，使用模拟数据
            if self._dry_run:
                self._logger.warning(f"[模拟模式] 获取持仓失败，使用空持仓: {e}")
                position = None
            else:
                raise

        if position:
            self._logger.info(
                f"当前持仓: {position.side} {position.size} @ {position.entry_price}, "
                f"未实现盈亏: {position.unrealized_pnl:+.2f} USDT"
            )

            # 检查持仓风险
            risk = self._risk_manager.check_position_risk(
                position=position,
                current_price=current_price,
            )

            if risk.action.value in ["REDUCE", "CLOSE"]:
                self._logger.warning(f"风险警报: {risk.reason}")
                self._notifier.send_alert(
                    title="持仓风险",
                    content=risk.reason,
                    severity="HIGH",
                )

                # 执行平仓
                if risk.action.value == "CLOSE" and not self._dry_run:
                    self._execute_close(position, risk.reason)
                    return
        else:
            self._logger.info("当前无持仓")

        # 4. AI 分析决策
        self._logger.debug("AI 分析中...")

        # 准备市场数据
        market_data = {
            "current_price": current_price,
            "change_24h": ticker.percentage,
            "volume_24h": ticker.quote_volume,
            "high_24h": ticker.high,
            "low_24h": ticker.low,
        }

        # 准备持仓数据
        position_data = None
        if position:
            position_data = {
                "side": position.side,
                "size": position.size,
                "entry_price": position.entry_price,
                "unrealized_pnl": position.unrealized_pnl,
            }

        # 调用 AI 分析
        ai_decision = self._analyzer.analyze(
            market_data=market_data,
            indicators=indicators,
            position=position_data,
        )

        self._logger.info(
            f"AI 决策: {ai_decision.decision.value} "
            f"(信心: {ai_decision.confidence.value})"
        )
        self._logger.info(f"决策理由: {ai_decision.reason}")

        # 5. 执行交易
        if ai_decision.is_actionable:
            # 检查交易冷却
            if self._check_trade_cooldown(ai_decision.decision.value, position):
                self._logger.info("交易冷却中，跳过本次操作")
            else:
                self._execute_decision(ai_decision, current_price, position)
        else:
            self._logger.debug("AI 建议观望，暂不操作")

    def _execute_decision(self, ai_decision, current_price: float, position):
        """执行 AI 决策

        Args:
            ai_decision: AI 决策
            current_price: 当前价格
            position: 当前持仓
        """
        symbol = get_config("trading.symbol")

        # 如果有持仓且决策是平仓
        if position and ai_decision.decision == Decision.CLOSE:
            if not self._dry_run:
                self._execute_close(position, ai_decision.reason)
            else:
                self._logger.info(f"[模拟] 平仓: {ai_decision.reason}")
            return

        # 如果没有持仓且决策是开仓
        if not position and ai_decision.decision in [Decision.BUY, Decision.SELL]:
            # 风险评估
            risk_assessment = self._risk_manager.assess_trade(
                symbol=symbol,
                side="long" if ai_decision.decision == Decision.BUY else "short",
                confidence=ai_decision.confidence.value,
                current_price=current_price,
            )

            if not risk_assessment.approved:
                self._logger.warning(f"交易被风险控制拒绝: {risk_assessment.message}")
                return

            # 计算仓位大小
            capital = self._position_manager.get_trading_capital()
            size = self._position_manager.calculate_size(
                capital=capital,
                price=current_price,
                confidence=ai_decision.confidence.value,
            )

            if not self._dry_run:
                self._execute_open(
                    side="buy" if ai_decision.decision == Decision.BUY else "sell",
                    size=size,
                    price=current_price,
                    stop_loss=ai_decision.stop_loss,
                    take_profit=ai_decision.take_profit,
                    reason=ai_decision.reason,
                )
            else:
                self._logger.info(
                    f"[模拟] 开仓: {ai_decision.decision.value} {size} @ {current_price}"
                )

    def _check_trade_cooldown(self, decision: str, position) -> bool:
        """检查交易冷却时间

        Args:
            decision: 决策类型 (BUY/SELL/CLOSE/HOLD)
            position: 当前持仓

        Returns:
            True 表示冷却中，应跳过交易；False 表示可以交易
        """
        now = datetime.now()

        # 获取冷却配置
        risk_config = get_config("risk", {})
        trade_cooldown = risk_config.get("trade_cooldown_seconds", 600)  # 默认10分钟
        position_cooldown = risk_config.get(
            "position_cooldown_seconds", 1800
        )  # 默认30分钟

        # 平仓后冷却检查
        if decision in ["BUY", "SELL"] and self._last_close_time:
            elapsed = (now - self._last_close_time).total_seconds()
            if elapsed < position_cooldown:
                remaining = int(position_cooldown - elapsed)
                self._logger.info(f"平仓后冷却中，还需等待 {remaining} 秒才能开新仓")
                return True

        # 同方向交易冷却检查
        if decision in ["BUY", "SELL"]:
            if self._last_trade_time and self._last_trade_side == decision:
                elapsed = (now - self._last_trade_time).total_seconds()
                if elapsed < trade_cooldown:
                    remaining = int(trade_cooldown - elapsed)
                    self._logger.info(f"{decision} 方向冷却中，还需等待 {remaining} 秒")
                    return True

        return False

    def _execute_open(
        self,
        side: str,
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "",
    ):
        """执行开仓

        Args:
            side: 方向 (buy/sell)
            size: 数量
            price: 价格
            stop_loss: 止损价
            take_profit: 止盈价
            reason: 原因
        """
        symbol = get_config("trading.symbol")
        leverage = get_config("trading.leverage", 10)

        self._logger.info(f"执行开仓: {side.upper()} {size} {symbol} @ {price}")

        try:
            # 设置杠杆
            self._exchange.set_leverage(leverage, symbol)

            # 下单
            result = self._exchange.create_market_order(
                symbol=symbol,
                side=side,
                size=size,
            )

            if result.success:
                self._trade_count += 1
                self._last_trade_time = datetime.now()
                self._last_trade_side = side.upper()
                self._logger.info(f"开仓成功: 订单ID {result.order_id}")

                # 发送通知
                self._notifier.send_trade(
                    action=side.upper(),
                    symbol=symbol,
                    price=price,
                    size=size,
                    confidence="",
                    reason=reason,
                )

                # TODO: 设置止损止盈订单
            else:
                self._logger.error(f"开仓失败: {result.error}")
                self._notifier.send_error("开仓失败", result.error or "Unknown error")

        except Exception as e:
            self._logger.error(f"开仓异常: {e}", exc_info=True)
            self._notifier.send_error("开仓异常", str(e))

    def _execute_close(self, position, reason: str = ""):
        """执行平仓

        Args:
            position: 持仓信息
            reason: 平仓原因
        """
        # 使用持仓的实际交易对，而不是配置中的交易对
        symbol = position.symbol

        self._logger.info(f"执行平仓: {position.side} {position.size} {symbol}")

        try:
            result = self._exchange.close_position(position=position)

            if result.success:
                self._trade_count += 1
                self._last_trade_time = datetime.now()
                self._last_close_time = datetime.now()  # 记录平仓时间
                self._logger.info(f"平仓成功: 订单ID {result.order_id}")

                # 更新本金追踪
                self._position_manager.update_tracking(
                    pnl=position.unrealized_pnl,
                    trade_info={"reason": reason},
                )

                # 发送通知
                self._notifier.send_trade(
                    action="CLOSE",
                    symbol=symbol,
                    price=position.entry_price,
                    size=position.size,
                    pnl=position.unrealized_pnl,
                    reason=reason,
                )
            else:
                self._logger.error(f"平仓失败: {result.error}")
                self._notifier.send_error("平仓失败", result.error or "Unknown error")

        except Exception as e:
            self._logger.error(f"平仓异常: {e}", exc_info=True)
            self._notifier.send_error("平仓异常", str(e))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PerpBot - AI 驱动的加密货币合约交易机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行模式（不下单）",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PerpBot {TradingBot.VERSION}",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(level=args.log_level)

    # 加载配置
    load_config(args.config)

    # 创建并启动机器人
    bot = TradingBot(dry_run=args.dry_run)
    bot.start()


if __name__ == "__main__":
    main()
