"""
data/market_data.py - Fetches OHLCV candles and computes all technical indicators.
"""

import logging
import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


class MarketData:
    """
    Fetches candle data from the exchange connector and enriches it with
    technical indicators used by the signal engine.
    """

    def __init__(self, connector, symbol: str, timeframe: str, limit: int = 200):
        self.connector = connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def fetch(self) -> pd.DataFrame:
        """
        Fetch latest OHLCV candles and compute all indicators.
        Returns a DataFrame with columns:
            timestamp, open, high, low, close, volume,
            rsi, macd, macd_signal, macd_hist,
            bb_upper, bb_middle, bb_lower, bb_pct,
            ema20, ema50,
            atr,
            volume_sma20, volume_ratio
        The last row is the most recent (potentially incomplete) candle.
        The second-to-last row is the last CLOSED candle — use this for signals.
        """
        raw = self.connector.fetch_ohlcv(self.symbol, self.timeframe, self.limit)
        if not raw or len(raw) < 60:
            raise RuntimeError(
                f"Not enough candle data returned ({len(raw)} candles). Need at least 60."
            )

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = self._add_indicators(df)
        logger.debug(
            f"MarketData fetched: {len(df)} candles | "
            f"Latest close: {df['close'].iloc[-1]:.4f} | "
            f"RSI: {df['rsi'].iloc[-2]:.1f}"
        )
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- RSI (14) ---
        rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
        df["rsi"] = rsi_indicator.rsi()

        # --- MACD (12, 26, 9) ---
        macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_hist"] = macd_indicator.macd_diff()

        # --- Bollinger Bands (20, 2) ---
        bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb_indicator.bollinger_hband()
        df["bb_middle"] = bb_indicator.bollinger_mavg()
        df["bb_lower"] = bb_indicator.bollinger_lband()
        # BB %B: where price sits within the bands (0=lower, 1=upper)
        df["bb_pct"] = bb_indicator.bollinger_pband()

        # --- EMA 20 and EMA 50 ---
        df["ema20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

        # --- ATR (14) ---
        df["atr"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        # --- Volume SMA (20) and ratio ---
        df["volume_sma20"] = volume.rolling(window=20).mean()
        df["volume_ratio"] = volume / df["volume_sma20"]

        # --- MACD crossover signals (for use in signal engine) ---
        # +1 = bullish cross (MACD crossed above signal), -1 = bearish cross, 0 = no cross
        df["macd_cross"] = 0
        macd_above = df["macd"] > df["macd_signal"]
        df.loc[macd_above & ~macd_above.shift(1).fillna(False), "macd_cross"] = 1
        df.loc[~macd_above & macd_above.shift(1).fillna(True), "macd_cross"] = -1

        return df

    def get_last_closed_candle(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the last FULLY closed candle (second-to-last row).
        The last row may be the current incomplete candle.
        """
        return df.iloc[-2]

    def get_current_candle(self, df: pd.DataFrame) -> pd.Series:
        """Returns the most recent candle (may be incomplete/live)."""
        return df.iloc[-1]

    def summary(self, df: pd.DataFrame) -> dict:
        """Returns a dict of key metrics from the last closed candle."""
        c = self.get_last_closed_candle(df)
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "close": round(float(c["close"]), 4),
            "rsi": round(float(c["rsi"]), 2),
            "macd": round(float(c["macd"]), 6),
            "macd_signal": round(float(c["macd_signal"]), 6),
            "macd_hist": round(float(c["macd_hist"]), 6),
            "macd_cross": int(c["macd_cross"]),
            "bb_upper": round(float(c["bb_upper"]), 4),
            "bb_lower": round(float(c["bb_lower"]), 4),
            "bb_pct": round(float(c["bb_pct"]), 4),
            "ema20": round(float(c["ema20"]), 4),
            "ema50": round(float(c["ema50"]), 4),
            "atr": round(float(c["atr"]), 4),
            "volume_ratio": round(float(c["volume_ratio"]), 2),
            "timestamp": str(c["timestamp"]),
        }
