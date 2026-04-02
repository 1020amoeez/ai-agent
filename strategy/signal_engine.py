"""
strategy/signal_engine.py - Multi-strategy scoring engine with market regime detection.

Combines 5 strategy modules into a unified confidence score (0-100):
  1. RSI + MACD Momentum        (25 pts)
  2. Bollinger Band Mean Reversion (15 pts)
  3. EMA Crossover Trend Following (20 pts)
  4. ADX Trend Strength           (15 pts)
  5. Volume Confirmation          (15 pts)
  + Multi-timeframe / Supertrend  (10 pts)

Market regime detection (ADX-based) auto-selects between trending and ranging strategies.
Dynamic SL/TP recommendations are included in the signal output (ATR-based).
"""

import logging
import math
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BUY  = "BUY"
SELL = "SELL"
HOLD = "HOLD"

# ── Indicator parameters ─────────────────────────────────────────────────────
RSI_OVERSOLD       = 30
RSI_OVERBOUGHT     = 70
STOCH_OVERSOLD     = 20
STOCH_OVERBOUGHT   = 80
ADX_TREND_THRESH   = 25
ADX_STRONG_THRESH  = 35

# ATR multipliers for dynamic SL/TP
SL_ATR_MULT_TIGHT  = 1.5   # high confidence
SL_ATR_MULT_STD    = 2.0   # standard
SL_ATR_MULT_WIDE   = 2.5   # low confidence / high volatility

TP_ATR_MULT_1      = 1.5   # first target
TP_ATR_MULT_2      = 2.5   # second target
TP_ATR_MULT_3      = 3.5   # runner target

# Confidence thresholds
CONF_HIGH   = 60
CONF_MEDIUM = 35   # lowered: agent was too conservative at 50
CONF_LOW    = 20


class SignalEngine:
    """
    Multi-strategy scoring engine.
    Evaluates the last closed candle across all strategies and returns a
    weighted confidence score that determines trade action and risk parameters.
    """

    def __init__(self):
        pass

    def evaluate(self, df: pd.DataFrame, in_position: bool = False) -> dict:
        """
        Main evaluation entry point.
        Returns signal dict with dynamic SL/TP recommendations.
        """
        if len(df) < 60:
            return self._signal(HOLD, "Not enough data for analysis.", 0, 0.0, df, {})

        c = df.iloc[-2]       # last closed candle
        prev = df.iloc[-3]    # candle before that

        price = float(c["close"])
        atr   = float(c["atr"]) if not pd.isna(c["atr"]) else 0.0

        # ── Detect market regime ─────────────────────────────────────────
        regime = self._detect_regime(c)

        # ── Score each strategy ──────────────────────────────────────────
        momentum_score, momentum_reasons = self._score_momentum(c, prev)
        mean_rev_score, mean_rev_reasons = self._score_mean_reversion(c, regime)
        trend_score, trend_reasons       = self._score_trend_following(c, prev, df)
        adx_score, adx_reasons           = self._score_adx_strength(c)
        volume_score, volume_reasons     = self._score_volume(c, df)
        bonus_score, bonus_reasons       = self._score_supertrend_bonus(c)

        # ── Weight scores by regime ──────────────────────────────────────
        if regime == "trending":
            # Favor trend-following and momentum in trending markets
            total_buy = (
                momentum_score["buy"] * 1.2 +
                mean_rev_score["buy"] * 0.5 +
                trend_score["buy"] * 1.3 +
                adx_score["buy"] * 1.2 +
                volume_score["buy"] * 1.0 +
                bonus_score["buy"] * 1.0
            )
            total_sell = (
                momentum_score["sell"] * 1.2 +
                mean_rev_score["sell"] * 0.5 +
                trend_score["sell"] * 1.3 +
                adx_score["sell"] * 1.2 +
                volume_score["sell"] * 1.0 +
                bonus_score["sell"] * 1.0
            )
        elif regime == "ranging":
            # Favor mean reversion in ranging markets
            total_buy = (
                momentum_score["buy"] * 0.8 +
                mean_rev_score["buy"] * 1.5 +
                trend_score["buy"] * 0.6 +
                adx_score["buy"] * 0.8 +
                volume_score["buy"] * 1.2 +
                bonus_score["buy"] * 0.8
            )
            total_sell = (
                momentum_score["sell"] * 0.8 +
                mean_rev_score["sell"] * 1.5 +
                trend_score["sell"] * 0.6 +
                adx_score["sell"] * 0.8 +
                volume_score["sell"] * 1.2 +
                bonus_score["sell"] * 0.8
            )
        else:
            # Neutral weighting
            total_buy = (
                momentum_score["buy"] +
                mean_rev_score["buy"] +
                trend_score["buy"] +
                adx_score["buy"] +
                volume_score["buy"] +
                bonus_score["buy"]
            )
            total_sell = (
                momentum_score["sell"] +
                mean_rev_score["sell"] +
                trend_score["sell"] +
                adx_score["sell"] +
                volume_score["sell"] +
                bonus_score["sell"]
            )

        # Normalize to 0-100
        buy_confidence  = min(int(total_buy), 100)
        sell_confidence = min(int(total_sell), 100)

        # Collect all reasons
        buy_reasons  = momentum_reasons["buy"] + mean_rev_reasons["buy"] + trend_reasons["buy"] + adx_reasons["buy"] + volume_reasons["buy"] + bonus_reasons["buy"]
        sell_reasons = momentum_reasons["sell"] + mean_rev_reasons["sell"] + trend_reasons["sell"] + adx_reasons["sell"] + volume_reasons["sell"] + bonus_reasons["sell"]

        # ── Calculate dynamic SL/TP based on ATR ─────────────────────────
        risk_params = self._calculate_risk_params(price, atr, buy_confidence, sell_confidence)

        # ── Generate signal ──────────────────────────────────────────────
        if in_position:
            # When in position, check sell signals
            if sell_confidence >= CONF_LOW:
                reason = f"SELL ({regime}) [{sell_confidence}%]: " + " | ".join(sell_reasons[:3])
                return self._signal(SELL, reason, sell_confidence, price, c, risk_params)
            return self._signal(HOLD, f"Holding position ({regime}). Buy={buy_confidence}% Sell={sell_confidence}%", 0, price, c, risk_params)
        else:
            # When not in position, check buy signals
            if buy_confidence >= CONF_MEDIUM:
                reason = f"BUY ({regime}) [{buy_confidence}%]: " + " | ".join(buy_reasons[:4])
                return self._signal(BUY, reason, buy_confidence, price, c, risk_params)
            return self._signal(HOLD, f"No signal ({regime}). Buy={buy_confidence}% Sell={sell_confidence}%", 0, price, c, risk_params)

    # ═════════════════════════════════════════════════════════════════════════
    # MARKET REGIME DETECTION
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_regime(self, c) -> str:
        """Detect market regime using ADX and Bollinger Band width."""
        adx = self._safe(c, "adx")

        if adx >= ADX_STRONG_THRESH:
            return "trending"
        elif adx >= ADX_TREND_THRESH:
            return "trending"
        else:
            return "ranging"

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: RSI + MACD MOMENTUM (max 25 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_momentum(self, c, prev) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        rsi       = self._safe(c, "rsi")
        macd_hist = self._safe(c, "macd_hist")
        macd_cross = self._safe(c, "macd_cross")
        stoch_k   = self._safe(c, "stoch_rsi_k")
        stoch_d   = self._safe(c, "stoch_rsi_d")
        prev_stoch_k = self._safe(prev, "stoch_rsi_k")
        prev_stoch_d = self._safe(prev, "stoch_rsi_d")

        # RSI: oversold/neutral = bullish opportunity
        if rsi < RSI_OVERSOLD:
            buy_score += 10
            buy_reasons.append(f"RSI oversold={rsi:.0f}")
        elif rsi < 40:
            buy_score += 7
            buy_reasons.append(f"RSI low={rsi:.0f}")
        elif rsi < 55:
            buy_score += 4
            buy_reasons.append(f"RSI neutral={rsi:.0f}")

        if rsi > RSI_OVERBOUGHT:
            sell_score += 12
            sell_reasons.append(f"RSI overbought={rsi:.0f}")
        elif rsi > 60:
            sell_score += 6
            sell_reasons.append(f"RSI high={rsi:.0f}")

        # MACD histogram momentum
        if macd_hist > 0:
            buy_score += 5
            buy_reasons.append(f"MACD hist+")
        else:
            sell_score += 5
            sell_reasons.append(f"MACD hist-")

        # MACD crossover
        if macd_cross == 1:
            buy_score += 7
            buy_reasons.append("MACD bullish cross")
        elif macd_cross == -1:
            sell_score += 8
            sell_reasons.append("MACD bearish cross")

        # Stochastic RSI crossover in extreme zones
        if stoch_k < STOCH_OVERSOLD and prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d:
            buy_score += 5
            buy_reasons.append(f"StochRSI bullish cross@{stoch_k:.0f}")
        elif stoch_k > STOCH_OVERBOUGHT and prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d:
            sell_score += 5
            sell_reasons.append(f"StochRSI bearish cross@{stoch_k:.0f}")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: BOLLINGER BAND MEAN REVERSION (max 15 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_mean_reversion(self, c, regime) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        bb_pct = self._safe(c, "bb_pct")
        rsi    = self._safe(c, "rsi")
        close  = self._safe(c, "close")
        bb_lower = self._safe(c, "bb_lower")
        bb_upper = self._safe(c, "bb_upper")

        # Price below lower BB + RSI oversold = strong mean reversion buy
        if bb_pct < 0.0:
            buy_score += 10
            buy_reasons.append(f"Below lower BB (BB%={bb_pct:.2f})")
        elif bb_pct < 0.2:
            buy_score += 6
            buy_reasons.append(f"Near lower BB (BB%={bb_pct:.2f})")
        elif bb_pct < 0.35 and rsi < 40:
            buy_score += 3
            buy_reasons.append(f"BB% low + RSI weak")

        # Price above upper BB + RSI overbought = mean reversion sell
        if bb_pct > 1.0:
            sell_score += 10
            sell_reasons.append(f"Above upper BB (BB%={bb_pct:.2f})")
        elif bb_pct > 0.8:
            sell_score += 6
            sell_reasons.append(f"Near upper BB (BB%={bb_pct:.2f})")
        elif bb_pct > 0.65 and rsi > 60:
            sell_score += 3
            sell_reasons.append(f"BB% high + RSI strong")

        # Extra points in ranging market
        if regime == "ranging":
            if bb_pct < 0.15 and rsi < 35:
                buy_score += 5
                buy_reasons.append("Range reversal setup")
            elif bb_pct > 0.85 and rsi > 65:
                sell_score += 5
                sell_reasons.append("Range reversal sell setup")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 3: EMA CROSSOVER TREND FOLLOWING (max 20 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_trend_following(self, c, prev, df) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        close   = self._safe(c, "close")
        ema9    = self._safe(c, "ema9")
        ema21   = self._safe(c, "ema21")
        ema55   = self._safe(c, "ema55")
        ema200  = self._safe(c, "ema200")
        ema_cross = self._safe(c, "ema_cross")

        # Macro trend: price vs EMA 200
        if close > ema200 and ema200 > 0:
            buy_score += 6
            buy_reasons.append("Above EMA200 (macro bullish)")
        elif close < ema200 and ema200 > 0:
            sell_score += 6
            sell_reasons.append("Below EMA200 (macro bearish)")

        # EMA alignment: 9 > 21 > 55 = strong uptrend
        if ema9 > ema21 > ema55 and ema55 > 0:
            buy_score += 5
            buy_reasons.append("EMAs aligned bullish (9>21>55)")
        elif ema9 < ema21 < ema55 and ema55 > 0:
            sell_score += 5
            sell_reasons.append("EMAs aligned bearish (9<21<55)")

        # EMA 9/21 crossover signal
        if ema_cross == 1:
            buy_score += 6
            buy_reasons.append("EMA 9/21 bullish cross")
        elif ema_cross == -1:
            sell_score += 6
            sell_reasons.append("EMA 9/21 bearish cross")

        # Price relative to EMA 20 (short-term trend)
        if close > ema21 and close > ema9:
            buy_score += 3
            buy_reasons.append("Price above short EMAs")
        elif close < ema21 and close < ema9:
            sell_score += 3
            sell_reasons.append("Price below short EMAs")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 4: ADX TREND STRENGTH (max 15 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_adx_strength(self, c) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        adx      = self._safe(c, "adx")
        plus_di  = self._safe(c, "plus_di")
        minus_di = self._safe(c, "minus_di")
        di_cross = self._safe(c, "di_cross")

        if adx < ADX_TREND_THRESH:
            # Weak trend - no directional score
            return {"buy": 0, "sell": 0}, {"buy": [], "sell": []}

        # ADX confirms trend exists
        trend_mult = 1.0
        if adx >= ADX_STRONG_THRESH:
            trend_mult = 1.3

        # +DI > -DI = bullish directional movement
        if plus_di > minus_di:
            score = int(8 * trend_mult)
            buy_score += score
            buy_reasons.append(f"ADX={adx:.0f} +DI>{'-'}DI (bullish)")
        else:
            score = int(8 * trend_mult)
            sell_score += score
            sell_reasons.append(f"ADX={adx:.0f} -DI>+DI (bearish)")

        # DI crossover
        if di_cross == 1:
            buy_score += 7
            buy_reasons.append("+DI bullish crossover")
        elif di_cross == -1:
            sell_score += 7
            sell_reasons.append("-DI bearish crossover")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # STRATEGY 5: VOLUME CONFIRMATION (max 15 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_volume(self, c, df) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        vol_ratio  = self._safe(c, "volume_ratio")
        close      = self._safe(c, "close")
        prev_close = float(df.iloc[-3]["close"]) if len(df) > 3 else close

        price_up = close > prev_close

        # Volume spike with price direction
        if vol_ratio >= 2.0:
            if price_up:
                buy_score += 10
                buy_reasons.append(f"Volume surge {vol_ratio:.1f}x + price up")
            else:
                sell_score += 10
                sell_reasons.append(f"Volume surge {vol_ratio:.1f}x + price down")
        elif vol_ratio >= 1.5:
            if price_up:
                buy_score += 7
                buy_reasons.append(f"Volume spike {vol_ratio:.1f}x + price up")
            else:
                sell_score += 7
                sell_reasons.append(f"Volume spike {vol_ratio:.1f}x + price down")
        elif vol_ratio >= 1.0:
            if price_up:
                buy_score += 3
                buy_reasons.append(f"Above avg volume")
            else:
                sell_score += 3

        # OBV trend confirmation (compare OBV direction to price direction over last 5 candles)
        if len(df) >= 7:
            obv_now  = float(df.iloc[-2]["obv"]) if not pd.isna(df.iloc[-2]["obv"]) else 0
            obv_prev = float(df.iloc[-7]["obv"]) if not pd.isna(df.iloc[-7]["obv"]) else 0
            price_now  = float(df.iloc[-2]["close"])
            price_prev = float(df.iloc[-7]["close"])

            obv_up = obv_now > obv_prev
            price_going_up = price_now > price_prev

            if obv_up and price_going_up:
                buy_score += 5
                buy_reasons.append("OBV confirms uptrend")
            elif not obv_up and not price_going_up:
                sell_score += 5
                sell_reasons.append("OBV confirms downtrend")
            elif obv_up and not price_going_up:
                buy_score += 3  # accumulation / bullish divergence
                buy_reasons.append("OBV bullish divergence")
            elif not obv_up and price_going_up:
                sell_score += 3  # distribution / bearish divergence
                sell_reasons.append("OBV bearish divergence")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # BONUS: SUPERTREND CONFIRMATION (max 10 pts)
    # ═════════════════════════════════════════════════════════════════════════

    def _score_supertrend_bonus(self, c) -> tuple:
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        st_dir = self._safe(c, "supertrend_dir")

        if st_dir == 1:
            buy_score += 10
            buy_reasons.append("Supertrend bullish")
        elif st_dir == -1:
            sell_score += 10
            sell_reasons.append("Supertrend bearish")

        return {"buy": buy_score, "sell": sell_score}, {"buy": buy_reasons, "sell": sell_reasons}

    # ═════════════════════════════════════════════════════════════════════════
    # DYNAMIC SL/TP CALCULATION
    # ═════════════════════════════════════════════════════════════════════════

    def _calculate_risk_params(self, price: float, atr: float,
                                buy_conf: int, sell_conf: int) -> dict:
        """
        Calculate dynamic stop-loss and take-profit based on ATR and confidence.
        Higher confidence = tighter stops (closer entry to ideal).
        Lower confidence = wider stops (more room for volatility).
        """
        if atr <= 0 or price <= 0:
            # Fallback to percentage-based
            return {
                "atr": 0,
                "sl_price": price * 0.98,
                "tp1_price": price * 1.03,
                "tp2_price": price * 1.05,
                "tp3_price": price * 1.07,
                "sl_atr_mult": SL_ATR_MULT_STD,
                "trailing_activate_atr": SL_ATR_MULT_STD,
                "trailing_distance_atr": SL_ATR_MULT_STD,
            }

        # Select ATR multiplier based on confidence
        conf = max(buy_conf, sell_conf)
        if conf >= CONF_HIGH:
            sl_mult = SL_ATR_MULT_TIGHT
        elif conf >= CONF_MEDIUM:
            sl_mult = SL_ATR_MULT_STD
        else:
            sl_mult = SL_ATR_MULT_WIDE

        return {
            "atr": round(atr, 6),
            "sl_price": round(price - (atr * sl_mult), 8),
            "tp1_price": round(price + (atr * TP_ATR_MULT_1), 8),
            "tp2_price": round(price + (atr * TP_ATR_MULT_2), 8),
            "tp3_price": round(price + (atr * TP_ATR_MULT_3), 8),
            "sl_atr_mult": sl_mult,
            # Trailing stop activates after price moves 1x ATR in favor
            "trailing_activate_atr": 1.0,
            # Trail distance tightens with confidence
            "trailing_distance_atr": sl_mult,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _safe(self, candle, key, default=0.0) -> float:
        """Safely extract a float value from a candle Series."""
        try:
            val = float(candle[key])
            if pd.isna(val) or math.isinf(val):
                return default
            return val
        except (KeyError, TypeError, ValueError):
            return default

    def _signal(self, signal: str, reason: str, confidence: int,
                price: float, candle, risk_params: dict) -> dict:
        rsi       = self._safe(candle, "rsi")
        macd_hist = self._safe(candle, "macd_hist")
        adx       = self._safe(candle, "adx")
        atr       = self._safe(candle, "atr")
        ema20     = self._safe(candle, "ema20")
        ema200    = self._safe(candle, "ema200")
        bb_pct    = self._safe(candle, "bb_pct")
        vol_ratio = self._safe(candle, "volume_ratio")
        stoch_k   = self._safe(candle, "stoch_rsi_k")
        st_dir    = self._safe(candle, "supertrend_dir")

        result = {
            "signal":     signal,
            "reason":     reason,
            "confidence": confidence,
            "price":      round(price, 4),
            "rsi":        round(rsi, 2),
            "macd_hist":  round(macd_hist, 6),
            "indicators": {
                "ema20":     round(ema20, 4),
                "ema200":    round(ema200, 4),
                "bb_pct":    round(bb_pct, 4),
                "vol_ratio": round(vol_ratio, 2),
                "adx":       round(adx, 2),
                "atr":       round(atr, 6),
                "stoch_k":   round(stoch_k, 2),
                "supertrend": int(st_dir),
            },
            "risk_params": risk_params,
        }
        logger.info(
            f"Signal: {signal:4s} | Confidence: {confidence:3d}% | "
            f"Price: {price:.4f} | RSI: {rsi:.1f} | ADX: {adx:.1f} | "
            f"Reason: {reason[:80]}"
        )
        return result
