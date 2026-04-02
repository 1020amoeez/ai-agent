"""
data/market_data.py - Fetches OHLCV candles and computes all technical indicators.
Enhanced with ADX, Stochastic RSI, multiple EMAs, and Supertrend for multi-strategy engine.
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
        The last row is the most recent (potentially incomplete) candle.
        The second-to-last row is the last CLOSED candle -- use this for signals.
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

        # --- Stochastic RSI (14, 3, 3) ---
        stoch_rsi = ta.momentum.StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k() * 100  # 0-100 scale
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d() * 100

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
        df["bb_pct"] = bb_indicator.bollinger_pband()

        # --- EMAs: 9, 20, 21, 50, 55, 200 ---
        df["ema9"]  = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        df["ema20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        df["ema21"] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        df["ema55"] = ta.trend.EMAIndicator(close=close, window=55).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

        # --- ADX (14) with +DI / -DI ---
        adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx_indicator.adx()
        df["plus_di"] = adx_indicator.adx_pos()
        df["minus_di"] = adx_indicator.adx_neg()

        # --- ATR (14) ---
        df["atr"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        # --- Volume SMA (20) and ratio ---
        df["volume_sma20"] = volume.rolling(window=20).mean()
        df["volume_ratio"] = volume / df["volume_sma20"]

        # --- OBV (On Balance Volume) ---
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        # --- MACD crossover signals ---
        df["macd_cross"] = 0
        macd_above = df["macd"] > df["macd_signal"]
        df.loc[macd_above & ~macd_above.shift(1).fillna(False), "macd_cross"] = 1
        df.loc[~macd_above & macd_above.shift(1).fillna(True), "macd_cross"] = -1

        # --- EMA crossover signals (9/21) ---
        df["ema_cross"] = 0
        ema_above = df["ema9"] > df["ema21"]
        df.loc[ema_above & ~ema_above.shift(1).fillna(False), "ema_cross"] = 1
        df.loc[~ema_above & ema_above.shift(1).fillna(True), "ema_cross"] = -1

        # --- DI crossover signals ---
        df["di_cross"] = 0
        di_above = df["plus_di"] > df["minus_di"]
        df.loc[di_above & ~di_above.shift(1).fillna(False), "di_cross"] = 1
        df.loc[~di_above & di_above.shift(1).fillna(True), "di_cross"] = -1

        # --- Supertrend (ATR 10, multiplier 3.0) ---
        df = self._add_supertrend(df, period=10, multiplier=3.0)

        return df

    def _add_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=period
        ).average_true_range()

        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 = uptrend, -1 = downtrend

        for i in range(1, len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                continue

            # Adjust bands based on previous values
            if lower_band.iloc[i] > lower_band.iloc[i-1] or df["close"].iloc[i-1] < lower_band.iloc[i-1]:
                pass  # keep current lower_band
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            if upper_band.iloc[i] < upper_band.iloc[i-1] or df["close"].iloc[i-1] > upper_band.iloc[i-1]:
                pass
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Determine direction
            if direction.iloc[i-1] == 1:  # was uptrend
                if df["close"].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
            else:  # was downtrend
                if df["close"].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]

        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction  # 1=bullish, -1=bearish

        return df

    def get_last_closed_candle(self, df: pd.DataFrame) -> pd.Series:
        """Returns the last FULLY closed candle (second-to-last row)."""
        return df.iloc[-2]

    def get_current_candle(self, df: pd.DataFrame) -> pd.Series:
        """Returns the most recent candle (may be incomplete/live)."""
        return df.iloc[-1]

    def summary(self, df: pd.DataFrame) -> dict:
        """Returns a dict of key metrics from the last closed candle."""
        c = self.get_last_closed_candle(df)

        def safe(val):
            v = float(val)
            if pd.isna(v):
                return 0.0
            return v

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "close": round(safe(c["close"]), 4),
            "rsi": round(safe(c["rsi"]), 2),
            "stoch_rsi_k": round(safe(c["stoch_rsi_k"]), 2),
            "stoch_rsi_d": round(safe(c["stoch_rsi_d"]), 2),
            "macd": round(safe(c["macd"]), 6),
            "macd_signal": round(safe(c["macd_signal"]), 6),
            "macd_hist": round(safe(c["macd_hist"]), 6),
            "macd_cross": int(safe(c["macd_cross"])),
            "bb_upper": round(safe(c["bb_upper"]), 4),
            "bb_lower": round(safe(c["bb_lower"]), 4),
            "bb_pct": round(safe(c["bb_pct"]), 4),
            "ema9": round(safe(c["ema9"]), 4),
            "ema20": round(safe(c["ema20"]), 4),
            "ema21": round(safe(c["ema21"]), 4),
            "ema50": round(safe(c["ema50"]), 4),
            "ema55": round(safe(c["ema55"]), 4),
            "ema200": round(safe(c["ema200"]), 4),
            "adx": round(safe(c["adx"]), 2),
            "plus_di": round(safe(c["plus_di"]), 2),
            "minus_di": round(safe(c["minus_di"]), 2),
            "atr": round(safe(c["atr"]), 4),
            "volume_ratio": round(safe(c["volume_ratio"]), 2),
            "supertrend_dir": int(safe(c["supertrend_dir"])),
            "ema_cross": int(safe(c["ema_cross"])),
            "di_cross": int(safe(c["di_cross"])),
            "timestamp": str(c["timestamp"]),
        }
