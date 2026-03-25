"""
main.py - AI Crypto Spot Trading Agent
Entry point. Runs the main agent loop.

Usage:
    python main.py

Requirements:
    Copy .env.example to .env and fill in your values before running.
"""

import sys
import time
import signal
import logging
import os
from datetime import datetime, date

# ── Setup logging before any imports that use it ────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"agent_{datetime.utcnow().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Suppress noisy library loggers
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger("main")

# ── Now import project modules ───────────────────────────────────────────────
try:
    from config import config, ConfigError
except ConfigError as e:
    logger.critical(f"Configuration error: {e}")
    sys.exit(1)

if config is None:
    logger.critical("Config failed to load. Check your .env file.")
    sys.exit(1)

from exchange.connector import ExchangeConnector
from data.market_data import MarketData
from strategy.signal_engine import SignalEngine
from risk.risk_manager import RiskManager
from execution.order_engine import OrderEngine
from notifications.telegram_bot import TelegramNotifier


# ── ANSI colours for console output ─────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BLUE   = "\033[94m"


def colored(text, color):
    return f"{color}{text}{C.RESET}"


# ── Graceful shutdown ────────────────────────────────────────────────────────
_running = True

def _shutdown_handler(signum, frame):
    global _running
    print(f"\n{colored('Shutdown signal received. Stopping agent gracefully...', C.YELLOW)}")
    _running = False

signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ── Display helpers ──────────────────────────────────────────────────────────

def print_header():
    print(colored("=" * 60, C.CYAN))
    print(colored("   AI CRYPTO SPOT TRADING AGENT", C.BOLD + C.WHITE))
    print(colored("=" * 60, C.CYAN))
    print(f"  Exchange : {colored(config.exchange.upper(), C.YELLOW)}")
    print(f"  Symbol   : {colored(config.symbol, C.YELLOW)}")
    print(f"  Timeframe: {colored(config.timeframe, C.YELLOW)}")
    print(f"  Mode     : {colored(config.trading_mode.upper(), C.GREEN if config.is_paper else C.RED)}")
    print(f"  Per Trade: {colored(f'${config.trade_amount_usdt}', C.YELLOW)}")
    print(f"  Stop Loss: {colored(f'{config.stop_loss_pct*100:.1f}%', C.RED)}")
    print(f"  Take Prof: {colored(f'{config.take_profit_pct*100:.1f}%', C.GREEN)}")
    print(colored("=" * 60, C.CYAN))


def print_status(iteration: int, market_summary: dict, signal: dict,
                 position_status: dict, balance: float, daily_pnl: float):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    sig = signal["signal"]
    sig_color = {
        "BUY":  C.GREEN,
        "SELL": C.RED,
        "HOLD": C.GREY,
    }.get(sig, C.WHITE)

    print()
    print(colored(f"── Iteration #{iteration}  {now} ──", C.CYAN))
    print(
        f"  Price   : {colored(f'${market_summary[\"close\"]:,.4f}', C.WHITE)}  "
        f"RSI: {colored(f'{market_summary[\"rsi\"]:.1f}', C.YELLOW)}  "
        f"EMA20: {colored(f'${market_summary[\"ema20\"]:,.2f}', C.BLUE)}  "
        f"Vol: {colored(f'{market_summary[\"volume_ratio\"]:.2f}x', C.GREY)}"
    )
    print(
        f"  MACD    : hist={colored(f'{market_summary[\"macd_hist\"]:+.6f}', C.GREEN if market_summary['macd_hist'] >= 0 else C.RED)}  "
        f"BB%: {colored(f'{market_summary[\"bb_pct\"]:.3f}', C.GREY)}"
    )
    print(
        f"  Signal  : {colored(sig, sig_color + C.BOLD)}  "
        f"Confidence: {colored(f'{signal[\"confidence\"]}%', C.YELLOW)}"
    )
    print(f"  Reason  : {colored(signal['reason'][:80], C.GREY)}")

    if position_status.get("in_position"):
        pnl     = position_status.get("unrealized_pnl", 0)
        pnl_pct = position_status.get("pnl_pct", 0)
        pnl_col = C.GREEN if pnl >= 0 else C.RED
        print(
            f"  Position: {colored('OPEN', C.GREEN)}  "
            f"Entry=${position_status['entry_price']:,.4f}  "
            f"P&L={colored(f'${pnl:+.2f} ({pnl_pct:+.2f}%)', pnl_col)}  "
            f"SL=${position_status['stop_loss']:,.4f}  "
            f"TP=${position_status['take_profit']:,.4f}"
        )
    else:
        print(f"  Position: {colored('NONE', C.GREY)}")

    pnl_col = C.GREEN if daily_pnl >= 0 else C.RED
    print(
        f"  Balance : {colored(f'${balance:,.2f} USDT', C.WHITE)}  "
        f"Daily P&L: {colored(f'${daily_pnl:+.2f}', pnl_col)}"
    )


def print_trade_event(side: str, price: float, amount: float, reason: str, pnl: float = None):
    if side == "BUY":
        bar = colored("▲ BUY  EXECUTED", C.GREEN + C.BOLD)
    else:
        bar = colored("▼ SELL EXECUTED", C.RED + C.BOLD)
    print(colored("  " + "─" * 56, C.CYAN))
    print(f"  {bar} | Price: ${price:,.4f} | Amount: {amount:.6f}")
    if pnl is not None:
        direction = colored(f"${pnl:+.2f}", C.GREEN if pnl >= 0 else C.RED)
        print(f"  P&L: {direction}")
    print(f"  Reason: {colored(reason[:80], C.GREY)}")
    print(colored("  " + "─" * 56, C.CYAN))


# ── Main loop ────────────────────────────────────────────────────────────────

def run():
    global _running

    print_header()
    logger.info("Agent starting up...")

    # ── Initialise all modules ───────────────────────────────────────────────
    notifier = TelegramNotifier(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )

    connector = ExchangeConnector(
        exchange_id=config.exchange,
        api_key=config.api_key,
        api_secret=config.api_secret,
        paper_mode=config.is_paper,
    )

    # Pre-load market info (validates symbol exists on exchange)
    if config.is_live:
        try:
            connector.load_markets(config.symbol)
        except Exception as e:
            logger.critical(f"Failed to load markets: {e}")
            sys.exit(1)

    market_data = MarketData(
        connector=connector,
        symbol=config.symbol,
        timeframe=config.timeframe,
        limit=config.candle_limit,
    )

    signal_engine = SignalEngine()

    risk_manager = RiskManager(
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        max_daily_loss_pct=config.max_daily_loss_pct,
        trade_amount_usdt=config.trade_amount_usdt,
    )

    order_engine = OrderEngine(
        connector=connector,
        risk_manager=risk_manager,
        notifier=notifier,
        symbol=config.symbol,
        paper_mode=config.is_paper,
    )

    notifier.send_startup(
        symbol=config.symbol,
        exchange=config.exchange,
        timeframe=config.timeframe,
        mode=config.trading_mode,
        trade_usdt=config.trade_amount_usdt,
    )

    logger.info(f"Agent ready. Loop interval: {config.loop_interval_seconds}s")
    print(colored(f"\nAgent running. Press Ctrl+C to stop.\n", C.GREEN))

    iteration      = 0
    last_summary_date = date.today()

    # ── Main loop ────────────────────────────────────────────────────────────
    while _running:
        iteration += 1

        try:
            # 1. Fetch market data + compute indicators
            df = market_data.fetch()
            summary = market_data.summary(df)

            # 2. Get current live price (for SL/TP monitoring, may differ from last candle close)
            try:
                current_price = connector.get_current_price(config.symbol)
            except Exception:
                current_price = summary["close"]

            # 3. Monitor open position for SL/TP hits (highest priority)
            if order_engine.is_in_position:
                exit_reason = order_engine.monitor_position(current_price)
                if exit_reason:
                    print_trade_event("SELL", current_price,
                                      risk_manager.open_position.amount_base if risk_manager.open_position else 0,
                                      exit_reason)

            # 4. Check kill switch
            balance = connector.get_free_usdt()
            portfolio_value = balance + (
                (risk_manager.open_position.amount_base * current_price)
                if risk_manager.is_in_position else 0
            )
            if risk_manager.check_kill_switch(portfolio_value):
                logger.warning("Kill switch active — skipping signal evaluation.")
                notifier.send_kill_switch(risk_manager.daily_pnl, portfolio_value)

            else:
                # 5. Generate signal from last closed candle
                signal = signal_engine.evaluate(df, in_position=order_engine.is_in_position)

                # 6. Act on signal
                if signal["signal"] == "BUY" and not order_engine.is_in_position:
                    notifier.send_signal(signal)
                    executed = order_engine.execute_buy(signal)
                    if executed and risk_manager.open_position:
                        pos = risk_manager.open_position
                        print_trade_event("BUY", pos.entry_price, pos.amount_base, signal["reason"])

                elif signal["signal"] == "SELL" and order_engine.is_in_position:
                    notifier.send_signal(signal)
                    pos = risk_manager.open_position
                    executed = order_engine.execute_sell(signal["reason"], current_price)
                    if executed and pos:
                        print_trade_event("SELL", current_price, pos.amount_base,
                                          signal["reason"], risk_manager.daily_pnl)

            # 7. Print status to console
            position_status = order_engine.get_position_status(current_price)
            print_status(
                iteration=iteration,
                market_summary=summary,
                signal=signal if not risk_manager.kill_switch_active else
                       {"signal": "HOLD", "reason": "Kill switch active", "confidence": 0,
                        "rsi": summary["rsi"], "macd_hist": summary["macd_hist"],
                        "indicators": {}},
                position_status=position_status,
                balance=balance,
                daily_pnl=risk_manager.daily_pnl,
            )

            # 8. Daily summary (send once per day at end of first loop after midnight)
            today = date.today()
            if today != last_summary_date:
                last_summary_date = today
                notifier.send_daily_summary({
                    "daily_pnl":         risk_manager.daily_pnl,
                    "trade_count_today": risk_manager.trade_count_today,
                    "balance_usdt":      balance,
                    "kill_switch":       risk_manager.kill_switch_active,
                })

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(f"Unhandled error in main loop (iteration {iteration}): {e}")
            notifier.send_error(f"Unhandled error (iteration {iteration}): {e}")
            print(colored(f"\n  [ERROR] {e} — resuming in 30s...\n", C.RED))
            time.sleep(30)
            continue

        # 9. Sleep until next candle
        if _running:
            sleep_secs = config.loop_interval_seconds
            print(colored(f"\n  Sleeping {sleep_secs}s until next candle...", C.GREY))
            # Sleep in small chunks so Ctrl+C is responsive
            for _ in range(sleep_secs):
                if not _running:
                    break
                time.sleep(1)

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Agent stopped.")
    print(colored("\nAgent stopped cleanly. Goodbye.", C.CYAN))

    # Final summary
    if risk_manager.trade_count_today > 0:
        notifier.send_daily_summary({
            "daily_pnl":         risk_manager.daily_pnl,
            "trade_count_today": risk_manager.trade_count_today,
            "balance_usdt":      connector.get_free_usdt(),
            "kill_switch":       risk_manager.kill_switch_active,
        })

    # Print trade history
    history = order_engine.get_trade_history()
    if history:
        print(colored("\n── Trade History ──", C.CYAN))
        for t in history:
            side_col = C.GREEN if t["side"] == "BUY" else C.RED
            pnl_str = f"  P&L: ${t['pnl']:+.2f}" if "pnl" in t else ""
            print(
                f"  {colored(t['side'], side_col)} {t['symbol']} "
                f"@ ${t['price']:,.4f}  qty={t['amount_base']:.6f}{pnl_str}"
            )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify .env exists
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print(colored(
            "\n[ERROR] .env file not found!\n"
            "  Copy .env.example to .env and fill in your settings:\n"
            "  cp .env.example .env\n",
            C.RED
        ))
        sys.exit(1)

    run()
