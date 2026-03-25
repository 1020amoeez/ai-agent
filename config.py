"""
config.py - Central configuration loader for the AI Trading Agent.
Loads all environment variables, validates them, and exposes a unified Config object.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

# All exchanges supported by ccxt (subset of most popular ones used in spot trading)
SUPPORTED_EXCHANGES = [
    "binance", "binanceus", "bybit", "okx", "kraken", "kucoin",
    "coinbase", "huobi", "gateio", "bitfinex", "bitstamp", "gemini",
    "mexc", "bitget", "phemex", "bitmex", "ftx", "ascendex",
    "bitmart", "bingx", "coinex", "digifinex", "exmo", "hitbtc",
    "lbank", "poloniex", "probit", "tidex", "upbit", "whitebit",
    "xt", "zb", "ace", "alpaca", "bequant", "bigone", "bitbns",
    "bitcoincom", "bitflyer", "bitpanda", "bitvavo", "bl3p",
    "btcmarkets", "cex", "coincheck", "coinsph", "cryptocom",
    "delta", "deribit", "dydx", "ellipx", "hollaex", "independentreserve",
    "indodax", "itbit", "latoken", "liquid", "mercado", "ndax",
    "novadax", "okcoin", "onetrading", "p2b", "paribu", "paymium",
    "ripio", "stex", "therock", "tidex", "timex", "tradeogre",
    "vcc", "wavesexchange", "wazirx", "yobit", "zaif"
]

VALID_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000
}


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class Config:
    """Loads, validates, and exposes all trading agent configuration."""

    def __init__(self):
        self._load()
        self._validate()

    def _get(self, key: str, default=None, required: bool = False):
        value = os.environ.get(key, default)
        if required and (value is None or str(value).strip() == ""):
            raise ConfigError(f"Required environment variable '{key}' is missing or empty.")
        return value

    def _load(self):
        # Exchange
        self.exchange = self._get("EXCHANGE", "binance").strip().lower()
        self.api_key = self._get("API_KEY", "").strip()
        self.api_secret = self._get("API_SECRET", "").strip()

        # Trading
        self.symbol = self._get("SYMBOL", "BTC/USDT").strip().upper()
        self.timeframe = self._get("TIMEFRAME", "5m").strip().lower()

        try:
            self.trade_amount_usdt = float(self._get("TRADE_AMOUNT_USDT", "50"))
        except (ValueError, TypeError):
            raise ConfigError("TRADE_AMOUNT_USDT must be a valid number.")

        # Risk Management
        try:
            self.stop_loss_pct = float(self._get("STOP_LOSS_PCT", "0.02"))
        except (ValueError, TypeError):
            raise ConfigError("STOP_LOSS_PCT must be a valid float (e.g. 0.02 for 2%).")

        try:
            self.take_profit_pct = float(self._get("TAKE_PROFIT_PCT", "0.04"))
        except (ValueError, TypeError):
            raise ConfigError("TAKE_PROFIT_PCT must be a valid float (e.g. 0.04 for 4%).")

        try:
            self.max_daily_loss_pct = float(self._get("MAX_DAILY_LOSS_PCT", "0.05"))
        except (ValueError, TypeError):
            raise ConfigError("MAX_DAILY_LOSS_PCT must be a valid float (e.g. 0.05 for 5%).")

        # Telegram (optional)
        self.telegram_bot_token = self._get("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = self._get("TELEGRAM_CHAT_ID", "").strip()
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)

        # Mode
        self.trading_mode = self._get("TRADING_MODE", "paper").strip().lower()

        # Derived: interval in seconds for the main loop
        self.loop_interval_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 300)

        # Number of candles to fetch for indicator calculations (need enough history)
        self.candle_limit = 200

    def _validate(self):
        errors = []

        if self.exchange not in SUPPORTED_EXCHANGES:
            errors.append(
                f"Exchange '{self.exchange}' is not in the supported list. "
                f"Supported exchanges: {', '.join(SUPPORTED_EXCHANGES[:20])} ... (and more ccxt exchanges)"
            )

        if self.timeframe not in VALID_TIMEFRAMES:
            errors.append(
                f"Timeframe '{self.timeframe}' is invalid. Valid options: {', '.join(VALID_TIMEFRAMES)}"
            )

        if "/" not in self.symbol:
            errors.append(f"SYMBOL '{self.symbol}' must be in BASE/QUOTE format (e.g. BTC/USDT).")

        if self.trade_amount_usdt <= 0:
            errors.append("TRADE_AMOUNT_USDT must be greater than 0.")

        if not (0 < self.stop_loss_pct < 1):
            errors.append("STOP_LOSS_PCT must be between 0 and 1 exclusive (e.g. 0.02 for 2%).")

        if not (0 < self.take_profit_pct < 1):
            errors.append("TAKE_PROFIT_PCT must be between 0 and 1 exclusive (e.g. 0.04 for 4%).")

        if not (0 < self.max_daily_loss_pct < 1):
            errors.append("MAX_DAILY_LOSS_PCT must be between 0 and 1 exclusive (e.g. 0.05 for 5%).")

        if self.stop_loss_pct >= self.take_profit_pct:
            errors.append("STOP_LOSS_PCT must be smaller than TAKE_PROFIT_PCT for a valid risk/reward ratio.")

        if self.trading_mode not in ("paper", "live"):
            errors.append("TRADING_MODE must be 'paper' or 'live'.")

        if self.trading_mode == "live" and (not self.api_key or not self.api_secret):
            errors.append("API_KEY and API_SECRET are required when TRADING_MODE=live.")

        if errors:
            raise ConfigError("Configuration errors found:\n  - " + "\n  - ".join(errors))

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.trading_mode == "live"

    def __repr__(self) -> str:
        return (
            f"Config("
            f"exchange={self.exchange}, "
            f"symbol={self.symbol}, "
            f"timeframe={self.timeframe}, "
            f"mode={self.trading_mode}, "
            f"trade_amount_usdt={self.trade_amount_usdt}, "
            f"stop_loss={self.stop_loss_pct*100:.1f}%, "
            f"take_profit={self.take_profit_pct*100:.1f}%, "
            f"max_daily_loss={self.max_daily_loss_pct*100:.1f}%, "
            f"telegram={'enabled' if self.telegram_enabled else 'disabled'}"
            f")"
        )


# Singleton config instance - import this from other modules
try:
    config = Config()
except ConfigError as e:
    print(f"[CONFIG ERROR] {e}", file=sys.stderr)
    # Don't exit here — allow partial imports during testing
    # The main.py will catch this on startup
    config = None
