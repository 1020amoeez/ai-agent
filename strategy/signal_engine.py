"""
strategy/signal_engine.py - Multi-confirmation signal engine.
Generates BUY / SELL / HOLD signals from indicator data.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Signal constants
BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"


class SignalEngine:
    """
    Evaluates technical indicators on the last closed candle and returns
    a structured signal dict.

    BUY conditions (all must be true for a strong buy):
        1. RSI < 45  — not overbought, room to grow
        2. MACD line crossed above signal line recently (last 3 candles)
        3. Price is above EMA20  — short-term uptrend
        4. Price is within 2% above or below the lower Bollinger Band
        5. Volume is above its 20-period SMA (volume confirmation)

    SELL conditions (any one triggers a sell):
        1. RSI > 70  — overbought
        2. MACD line crossed below signal line
        3. Price drops below EMA20 AND RSI < 45 (trend reversal)
        4. Price hits take-profit level (managed by RiskManager/OrderEngine)
        5. Price hits stop-loss level  (managed by RiskManager/OrderEngine)

    Confidence score (0–100): number of BUY sub-conditions met × 20.
    """

    def __init__(self, rsi_buy_threshold: float = 45.0, rsi_sell_threshold: float = 70.0):
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    def evaluate(self, df: pd.DataFrame, in_position: bool = False) -> dict:
        """
        Evaluate the last closed candle (df.iloc[-2]) and return a signal.

        Args:
            df: DataFrame with all indicators computed by MarketData.
            in_position: True if the agent currently holds a position.

        Returns:
            {
                "signal":     "BUY" | "SELL" | "HOLD",
                "reason":     str,
                "confidence": int (0-100),
                "price":      float,
                "rsi":        float,
                "macd_hist":  float,
                "indicators": dict  (full snapshot)
            }
        """
        if len(df) < 3:
            return self._signal(HOLD, "Not enough candle data.", 0, 0.0, df)

        # Use the last CLOSED candle for signal generation
        c = df.iloc[-2]

        price      = float(c["close"])
        rsi        = float(c["rsi"])
        macd       = float(c["macd"])
        macd_sig   = float(c["macd_signal"])
        macd_hist  = float(c["macd_hist"])
        macd_cross = int(c["macd_cross"])
        bb_lower   = float(c["bb_lower"])
        bb_upper   = float(c["bb_upper"])
        bb_pct     = float(c["bb_pct"])
        ema20      = float(c["ema20"])
        ema50      = float(c["ema50"])
        vol_ratio  = float(c["volume_ratio"])

        # ---- SELL checks (take priority when in position) ----
        if in_position:
            sell_reason = self._check_sell_conditions(
                rsi, macd, macd_sig, macd_cross, price, ema20
            )
            if sell_reason:
                return self._signal(SELL, sell_reason, 100, price, c)

        # ---- BUY checks (only when not in position) ----
        if not in_position:
            buy_signal, confidence, buy_reason = self._check_buy_conditions(
                rsi, macd, macd_sig, macd_cross, price, bb_lower, ema20, vol_ratio
            )
            if buy_signal:
                return self._signal(BUY, buy_reason, confidence, price, c)

        return self._signal(HOLD, "No actionable signal.", 0, price, c)

    # -------------------------------------------------------------------------
    # BUY logic
    # -------------------------------------------------------------------------

    def _check_buy_conditions(
        self, rsi, macd, macd_sig, macd_cross, price, bb_lower, ema20, vol_ratio
    ):
        """
        Checks all BUY sub-conditions.
        Returns (is_buy: bool, confidence: int, reason: str).
        """
        conditions = []
        reasons = []

        # Condition 1: RSI not overbought
        cond1 = rsi < self.rsi_buy_threshold
        conditions.append(cond1)
        if cond1:
            reasons.append(f"RSI={rsi:.1f}<{self.rsi_buy_threshold}")

        # Condition 2: MACD bullish cross in last 3 candles
        cond2 = macd_cross == 1 or (macd > macd_sig and macd_cross >= 0)
        conditions.append(cond2)
        if cond2:
            reasons.append("MACD bullish")

        # Condition 3: Price above EMA20
        cond3 = price > ema20
        conditions.append(cond3)
        if cond3:
            reasons.append(f"Price>{ema20:.2f}(EMA20)")

        # Condition 4: Price near lower Bollinger Band (within 2% above lower band)
        lower_band_zone = bb_lower * 1.02
        cond4 = price <= lower_band_zone
        conditions.append(cond4)
        if cond4:
            reasons.append(f"Near BB lower({bb_lower:.2f})")

        # Condition 5: Volume confirmation
        cond5 = vol_ratio >= 1.0
        conditions.append(cond5)
        if cond5:
            reasons.append(f"Vol ratio={vol_ratio:.2f}x")

        met = sum(conditions)
        confidence = met * 20  # 5 conditions × 20 = 100 max

        # Require at least 3 of 5 conditions for a BUY signal
        if met >= 3 and cond1 and cond2:
            reason = " | ".join(reasons)
            return True, confidence, f"BUY signal ({met}/5 conditions): {reason}"

        return False, confidence, ""

    # -------------------------------------------------------------------------
    # SELL logic
    # -------------------------------------------------------------------------

    def _check_sell_conditions(self, rsi, macd, macd_sig, macd_cross, price, ema20):
        """
        Checks SELL conditions. Returns reason string or None.
        SL/TP are handled separately by the OrderEngine.
        """

        # Condition 1: RSI overbought
        if rsi > self.rsi_sell_threshold:
            return f"RSI overbought ({rsi:.1f} > {self.rsi_sell_threshold})"

        # Condition 2: MACD bearish cross
        if macd_cross == -1:
            return f"MACD bearish crossover (hist={macd - macd_sig:.6f})"

        # Condition 3: MACD below signal AND price dropped below EMA20
        if macd < macd_sig and price < ema20:
            return f"MACD negative + Price below EMA20 ({ema20:.2f})"

        return None

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _signal(self, signal: str, reason: str, confidence: int, price: float, candle) -> dict:
        try:
            rsi       = round(float(candle["rsi"]), 2)
            macd_hist = round(float(candle["macd_hist"]), 6)
            ema20     = round(float(candle["ema20"]), 4)
            bb_pct    = round(float(candle["bb_pct"]), 4)
            vol_ratio = round(float(candle["volume_ratio"]), 2)
        except (KeyError, TypeError):
            rsi = macd_hist = ema20 = bb_pct = vol_ratio = 0.0

        result = {
            "signal":     signal,
            "reason":     reason,
            "confidence": confidence,
            "price":      round(price, 4),
            "rsi":        rsi,
            "macd_hist":  macd_hist,
            "indicators": {
                "ema20":      ema20,
                "bb_pct":     bb_pct,
                "vol_ratio":  vol_ratio,
            },
        }
        logger.info(
            f"Signal: {signal:4s} | Confidence: {confidence:3d}% | "
            f"Price: {price:.4f} | RSI: {rsi:.1f} | Reason: {reason}"
        )
        return result
