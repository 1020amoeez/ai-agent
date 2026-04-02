"""
risk/risk_manager.py - Dynamic ATR-based risk management with trailing stops.

Replaces fixed percentage SL/TP with:
  - ATR-based dynamic stop loss and take profit
  - Chandelier-style trailing stop that locks in profits
  - Breakeven stop after price moves 1x ATR in favor
  - Confidence-based position sizing
"""
from __future__ import annotations
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class Position:
    """Represents a currently open trading position with dynamic risk levels."""

    def __init__(self, entry_price: float, amount_base: float, amount_usdt: float,
                 stop_loss: float, take_profit: float, atr: float = 0.0,
                 trailing_activate_atr: float = 1.0, trailing_distance_atr: float = 2.0):
        self.entry_price   = entry_price
        self.amount_base   = amount_base
        self.amount_usdt   = amount_usdt
        self.stop_loss     = stop_loss       # dynamic, updated by trailing
        self.take_profit   = take_profit     # TP1 target
        self.tp2           = 0.0             # TP2 target
        self.tp3           = 0.0             # TP3 target
        self.atr           = atr
        self.opened_at     = datetime.utcnow()
        self.order_id      = None
        self.highest_price = entry_price     # track highest price for trailing stop
        self.breakeven_hit = False           # has price reached breakeven activation?
        self.trailing_active = False         # is trailing stop active?
        self.trailing_activate_atr = trailing_activate_atr
        self.trailing_distance_atr = trailing_distance_atr
        self.initial_stop_loss = stop_loss   # keep original SL for reference

    def unrealized_pnl(self, current_price: float) -> float:
        current_value = self.amount_base * current_price
        return current_value - self.amount_usdt

    def pnl_pct(self, current_price: float) -> float:
        if self.amount_usdt == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.amount_usdt) * 100

    def update_trailing_stop(self, current_price: float):
        """
        Update trailing stop based on current price.
        - After price moves 1x ATR above entry: move stop to breakeven
        - After that: trail stop at trailing_distance_atr below highest price
        - Stop only moves UP, never down
        """
        if self.atr <= 0:
            return

        # Track highest price
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Phase 1: Breakeven stop
        # When price moves 1x ATR above entry, move stop to breakeven + small buffer
        breakeven_trigger = self.entry_price + (self.atr * self.trailing_activate_atr)
        if not self.breakeven_hit and current_price >= breakeven_trigger:
            self.breakeven_hit = True
            self.trailing_active = True
            new_sl = self.entry_price + (self.atr * 0.1)  # just above breakeven
            if new_sl > self.stop_loss:
                self.stop_loss = round(new_sl, 8)
                logger.info(f"Breakeven stop activated: SL moved to ${self.stop_loss:.4f}")

        # Phase 2: Chandelier trailing stop
        # Trail at trailing_distance_atr below the highest price reached
        if self.trailing_active:
            trail_sl = self.highest_price - (self.atr * self.trailing_distance_atr)
            if trail_sl > self.stop_loss:
                self.stop_loss = round(trail_sl, 8)
                logger.debug(f"Trailing stop updated: SL=${self.stop_loss:.4f} (highest=${self.highest_price:.4f})")

            # Phase 3: Tighten trail after 2x ATR profit
            if current_price >= self.entry_price + (self.atr * 2.5):
                tight_sl = self.highest_price - (self.atr * 1.0)
                if tight_sl > self.stop_loss:
                    self.stop_loss = round(tight_sl, 8)
                    logger.debug(f"Tight trailing stop: SL=${self.stop_loss:.4f}")

    def __repr__(self):
        return (
            f"Position(entry={self.entry_price:.4f}, base={self.amount_base:.6f}, "
            f"usdt={self.amount_usdt:.2f}, sl={self.stop_loss:.4f}, tp={self.take_profit:.4f}, "
            f"atr={self.atr:.4f}, trailing={'ON' if self.trailing_active else 'OFF'})"
        )


class RiskManager:
    """
    Dynamic ATR-based risk management.
    - Position sizing based on ATR and risk per trade
    - Dynamic SL/TP from signal engine's risk_params
    - Trailing stop management
    - Daily P&L tracking and kill switch
    """

    MAX_POSITION_SIZE_PCT = 0.25  # max 25% of portfolio per trade

    def __init__(self, max_daily_loss_pct: float, trade_amount_usdt: float,
                 risk_per_trade_pct: float = 0.02):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.trade_amount_usdt  = trade_amount_usdt
        self.risk_per_trade_pct = risk_per_trade_pct  # 2% default

        self._daily_pnl         = 0.0
        self._daily_pnl_date    = date.today()
        self._kill_switch_active = False
        self._trade_count_today  = 0
        self.open_position: Position | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Trade Approval
    # ─────────────────────────────────────────────────────────────────────────

    def approve_trade(self, signal: str, free_usdt: float, current_price: float) -> tuple[bool, str]:
        self._reset_daily_if_needed()

        if self._kill_switch_active:
            return False, f"Kill switch active -- daily loss limit reached ({self.max_daily_loss_pct*100:.1f}%)"
        if signal != "BUY":
            return False, f"Signal is {signal}, not BUY."
        if self.open_position is not None:
            return False, "Already in position."
        if free_usdt < 10:
            return False, f"Insufficient balance (${free_usdt:.2f} < $10)."

        trade_usdt = self._calculate_trade_usdt(free_usdt)
        if trade_usdt <= 0:
            return False, "Calculated trade size is zero."

        return True, f"Trade approved. Size: ${trade_usdt:.2f} USDT"

    # ─────────────────────────────────────────────────────────────────────────
    # Position Sizing (ATR-aware)
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_trade_usdt(self, free_usdt: float, atr: float = 0, price: float = 0,
                              sl_atr_mult: float = 2.0) -> float:
        """
        Calculate trade size. If ATR is available, size based on risk:
        position_size = (account * risk%) / (ATR * sl_multiplier)
        Otherwise falls back to fixed amount.
        """
        if atr > 0 and price > 0:
            risk_amount = free_usdt * self.risk_per_trade_pct
            risk_per_unit = atr * sl_atr_mult
            if risk_per_unit > 0:
                optimal_units = risk_amount / risk_per_unit
                optimal_usdt = optimal_units * price
                # Cap at configured amount and max position size
                trade_usdt = min(optimal_usdt, self.trade_amount_usdt,
                                 free_usdt * self.MAX_POSITION_SIZE_PCT)
                return round(max(trade_usdt, 0.0), 2)

        return self._calculate_trade_usdt(free_usdt)

    def _calculate_trade_usdt(self, free_usdt: float) -> float:
        max_allowed = free_usdt * self.MAX_POSITION_SIZE_PCT
        trade_usdt  = min(self.trade_amount_usdt, max_allowed)
        return round(max(trade_usdt, 0.0), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic SL/TP from Signal Engine
    # ─────────────────────────────────────────────────────────────────────────

    def get_dynamic_sl_tp(self, entry_price: float, risk_params: dict) -> dict:
        """
        Extract dynamic SL/TP from signal engine's risk_params.
        Falls back to 2% SL / 4% TP if no ATR data.
        """
        if risk_params and risk_params.get("atr", 0) > 0:
            return {
                "stop_loss":   risk_params["sl_price"],
                "take_profit": risk_params["tp1_price"],
                "tp2":         risk_params["tp2_price"],
                "tp3":         risk_params["tp3_price"],
                "atr":         risk_params["atr"],
                "sl_atr_mult": risk_params.get("sl_atr_mult", 2.0),
                "trailing_activate_atr": risk_params.get("trailing_activate_atr", 1.0),
                "trailing_distance_atr": risk_params.get("trailing_distance_atr", 2.0),
            }

        # Fallback: percentage-based
        return {
            "stop_loss":   round(entry_price * 0.98, 8),
            "take_profit": round(entry_price * 1.04, 8),
            "tp2":         round(entry_price * 1.06, 8),
            "tp3":         round(entry_price * 1.08, 8),
            "atr":         0.0,
            "sl_atr_mult": 2.0,
            "trailing_activate_atr": 1.0,
            "trailing_distance_atr": 2.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Position Management
    # ─────────────────────────────────────────────────────────────────────────

    def open_position_record(self, entry_price: float, amount_base: float,
                              amount_usdt: float, risk_params: dict = None,
                              order_id: str = None) -> Position:
        """Records a new open position with dynamic ATR-based levels."""
        levels = self.get_dynamic_sl_tp(entry_price, risk_params or {})

        pos = Position(
            entry_price=entry_price,
            amount_base=amount_base,
            amount_usdt=amount_usdt,
            stop_loss=levels["stop_loss"],
            take_profit=levels["take_profit"],
            atr=levels["atr"],
            trailing_activate_atr=levels["trailing_activate_atr"],
            trailing_distance_atr=levels["trailing_distance_atr"],
        )
        pos.tp2 = levels["tp2"]
        pos.tp3 = levels["tp3"]
        pos.order_id = order_id
        self.open_position = pos
        self._trade_count_today += 1

        sl_dist = abs(entry_price - levels["stop_loss"])
        tp_dist = abs(levels["take_profit"] - entry_price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        logger.info(
            f"Position opened: entry=${entry_price:.4f} | "
            f"SL=${levels['stop_loss']:.4f} (ATR×{levels['sl_atr_mult']:.1f}) | "
            f"TP1=${levels['take_profit']:.4f} | TP2=${levels['tp2']:.4f} | "
            f"R:R=1:{rr:.1f} | ATR=${levels['atr']:.4f}"
        )
        return pos

    def close_position_record(self, exit_price: float) -> float:
        if self.open_position is None:
            return 0.0

        pnl = self.open_position.unrealized_pnl(exit_price)
        self._update_daily_pnl(pnl)

        logger.info(
            f"Position closed: exit=${exit_price:.4f} | "
            f"P&L=${pnl:+.2f} | Daily P&L=${self._daily_pnl:+.2f}"
        )
        self.open_position = None
        return pnl

    # ─────────────────────────────────────────────────────────────────────────
    # SL/TP Monitoring with Trailing Stop
    # ─────────────────────────────────────────────────────────────────────────

    def update_trailing_stop(self, current_price: float):
        """Update the trailing stop for the open position."""
        if self.open_position:
            self.open_position.update_trailing_stop(current_price)

    def check_stop_loss(self, current_price: float) -> bool:
        if self.open_position is None:
            return False
        hit = current_price <= self.open_position.stop_loss
        if hit:
            sl_type = "trailing" if self.open_position.trailing_active else "initial"
            logger.warning(
                f"STOP LOSS HIT ({sl_type}): price={current_price:.4f} <= "
                f"sl={self.open_position.stop_loss:.4f}"
            )
        return hit

    def check_take_profit(self, current_price: float) -> bool:
        if self.open_position is None:
            return False
        hit = current_price >= self.open_position.take_profit
        if hit:
            logger.info(
                f"TAKE PROFIT HIT: price={current_price:.4f} >= "
                f"tp={self.open_position.take_profit:.4f}"
            )
        return hit

    # ─────────────────────────────────────────────────────────────────────────
    # Daily P&L & Kill Switch
    # ─────────────────────────────────────────────────────────────────────────

    def _update_daily_pnl(self, pnl: float):
        self._reset_daily_if_needed()
        self._daily_pnl += pnl
        logger.info(f"Daily P&L updated: ${self._daily_pnl:+.2f}")

    def check_kill_switch(self, portfolio_value: float) -> bool:
        self._reset_daily_if_needed()
        if self._kill_switch_active:
            return True

        if portfolio_value > 0:
            loss_pct = -self._daily_pnl / portfolio_value
            if loss_pct >= self.max_daily_loss_pct:
                self._kill_switch_active = True
                logger.critical(
                    f"KILL SWITCH ACTIVATED: daily loss {loss_pct*100:.2f}% "
                    f">= limit {self.max_daily_loss_pct*100:.1f}%"
                )
        return self._kill_switch_active

    def reset_kill_switch(self):
        self._kill_switch_active = False
        logger.info("Kill switch manually reset.")

    def _reset_daily_if_needed(self):
        today = date.today()
        if today != self._daily_pnl_date:
            logger.info(f"New trading day ({today}). Resetting daily P&L from ${self._daily_pnl:+.2f}")
            self._daily_pnl = 0.0
            self._daily_pnl_date = today
            self._kill_switch_active = False
            self._trade_count_today = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_in_position(self) -> bool:
        return self.open_position is not None

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trade_count_today(self) -> int:
        return self._trade_count_today

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def get_stats(self) -> dict:
        return {
            "daily_pnl":         round(self._daily_pnl, 2),
            "trade_count_today": self._trade_count_today,
            "kill_switch":       self._kill_switch_active,
            "in_position":       self.is_in_position,
            "open_position":     repr(self.open_position) if self.open_position else None,
        }
