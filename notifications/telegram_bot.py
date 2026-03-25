"""
notifications/telegram_bot.py - Sends trade alerts and summaries via Telegram Bot API.
No extra library needed — uses requests directly.
Silently skips all notifications if token/chat_id are not configured.
"""

import logging
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Sends formatted Telegram messages for:
    - BUY / SELL signals
    - Trade execution confirmations
    - Error alerts
    - Daily P&L summary
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id   = chat_id
        self.enabled   = bool(bot_token and chat_id)

        if self.enabled:
            logger.info("Telegram notifier enabled.")
        else:
            logger.info("Telegram notifier disabled (no token/chat_id configured).")

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def send_signal(self, signal: dict):
        """Sends a formatted signal alert."""
        if not self.enabled:
            return

        s      = signal.get("signal", "?")
        price  = signal.get("price", 0)
        rsi    = signal.get("rsi", 0)
        conf   = signal.get("confidence", 0)
        reason = signal.get("reason", "")
        inds   = signal.get("indicators", {})

        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪️"}.get(s, "❓")

        text = (
            f"{emoji} *{s} SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💲 Price:       `${price:,.4f}`\n"
            f"📊 RSI:         `{rsi:.1f}`\n"
            f"📈 EMA20:       `${inds.get('ema20', 0):,.4f}`\n"
            f"📉 BB%:         `{inds.get('bb_pct', 0):.3f}`\n"
            f"🔊 Vol Ratio:   `{inds.get('vol_ratio', 0):.2f}x`\n"
            f"🎯 Confidence:  `{conf}%`\n"
            f"📝 Reason: _{reason}_"
        )
        self._send(text)

    def send_trade_executed(
        self, side: str, symbol: str, price: float, amount: float,
        trade_usdt: float = 0, stop_loss: float = None, take_profit: float = None,
        pnl: float = None, pnl_pct: float = None, reason: str = "", paper: bool = True
    ):
        """Sends a trade execution confirmation."""
        if not self.enabled:
            return

        mode_tag = "📋 PAPER" if paper else "⚡ LIVE"
        emoji    = "🟢 BUY" if side == "BUY" else "🔴 SELL"

        lines = [
            f"{emoji} *ORDER EXECUTED* {mode_tag}",
            f"━━━━━━━━━━━━━━━━━━",
            f"🪙 Symbol:   `{symbol}`",
            f"💲 Price:    `${price:,.4f}`",
            f"📦 Amount:   `{amount:.6f}`",
            f"💵 Value:    `${trade_usdt:,.2f} USDT`",
        ]

        if stop_loss:
            lines.append(f"🛑 Stop Loss: `${stop_loss:,.4f}`")
        if take_profit:
            lines.append(f"🎯 Take Profit: `${take_profit:,.4f}`")
        if pnl is not None:
            direction = "✅ PROFIT" if pnl >= 0 else "❌ LOSS"
            lines.append(f"{direction}: `${pnl:+,.2f}` (`{pnl_pct:+.2f}%`)")
        if reason:
            lines.append(f"📝 _{reason}_")

        self._send("\n".join(lines))

    def send_error(self, error_msg: str):
        """Sends an error alert."""
        if not self.enabled:
            return
        text = f"🚨 *AGENT ERROR*\n━━━━━━━━━━━━\n`{error_msg}`"
        self._send(text)

    def send_kill_switch(self, daily_pnl: float, portfolio_value: float):
        """Sends a kill switch activation alert."""
        if not self.enabled:
            return
        pct = abs(daily_pnl / portfolio_value * 100) if portfolio_value else 0
        text = (
            f"🚫 *KILL SWITCH ACTIVATED*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Daily loss: `${daily_pnl:,.2f}` (`{pct:.1f}%`)\n"
            f"Trading halted for today."
        )
        self._send(text)

    def send_daily_summary(self, stats: dict):
        """Sends a daily P&L summary."""
        if not self.enabled:
            return

        pnl         = stats.get("daily_pnl", 0)
        trades      = stats.get("trade_count_today", 0)
        balance     = stats.get("balance_usdt", 0)
        kill_switch = stats.get("kill_switch", False)
        direction   = "✅ Profit" if pnl >= 0 else "❌ Loss"

        text = (
            f"📅 *DAILY SUMMARY*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Trades today: `{trades}`\n"
            f"{direction}: `${pnl:+,.2f}`\n"
            f"💰 Balance: `${balance:,.2f} USDT`\n"
            f"🔒 Kill switch: `{'ON' if kill_switch else 'OFF'}`"
        )
        self._send(text)

    def send_startup(self, symbol: str, exchange: str, timeframe: str,
                     mode: str, trade_usdt: float):
        """Sends a startup notification."""
        if not self.enabled:
            return
        mode_emoji = "📋" if mode == "paper" else "⚡"
        text = (
            f"🤖 *AI TRADING AGENT STARTED*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏦 Exchange: `{exchange}`\n"
            f"🪙 Symbol:   `{symbol}`\n"
            f"⏱ Timeframe: `{timeframe}`\n"
            f"💵 Per trade: `${trade_usdt}`\n"
            f"{mode_emoji} Mode: `{mode.upper()}`"
        )
        self._send(text)

    def send_message(self, text: str):
        """Send a raw message."""
        if not self.enabled:
            return
        self._send(text)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _send(self, text: str):
        """Sends a message via the Telegram Bot API."""
        url = TELEGRAM_API.format(token=self.bot_token)
        payload = {
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if not response.ok:
                logger.warning(
                    f"Telegram send failed: {response.status_code} — {response.text[:200]}"
                )
            else:
                logger.debug("Telegram message sent successfully.")
        except requests.exceptions.Timeout:
            logger.warning("Telegram send timed out.")
        except requests.exceptions.ConnectionError:
            logger.warning("Telegram send failed — no network connection.")
        except Exception as e:
            logger.warning(f"Telegram send unexpected error: {e}")
