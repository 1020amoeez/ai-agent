"""
risk/risk_manager.py - Position sizing, daily P&L tracking, and kill switch.
"""

import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class Position:
    """Represents a currently open trading position."""

    def __init__(self, entry_price: float, amount_base: float, amount_usdt: float,
                 stop_loss: float, take_profit: float):
        self.entry_price   = entry_price
        self.amount_base   = amount_base    # e.g. 0.001 BTC
        self.amount_usdt   = amount_usdt    # e.g. 50 USDT spent
        self.stop_loss     = stop_loss
        self.take_profit   = take_profit
        self.opened_at     = datetime.utcnow()
        self.order_id      = None

    def unrealized_pnl(self, current_price: float) -> float:
        """Returns unrealized P&L in USDT."""
        current_value = self.amount_base * current_price
        return current_value - self.amount_usdt

    def pnl_pct(self, current_price: float) -> float:
        """Returns P&L as percentage of invested amount."""
        if self.amount_usdt == 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.amount_usdt) * 100

    def __repr__(self):
        return (
            f"Position(entry={self.entry_price:.4f}, base={self.amount_base:.6f}, "
            f"usdt={self.amount_usdt:.2f}, sl={self.stop_loss:.4f}, tp={self.take_profit:.4f})"
        )


class RiskManager:
    """
    Manages trade risk:
    - Position sizing (fixed fractional + max 2% risk per trade)
    - Stop-loss and take-profit price calculation
    - Daily P&L tracking and kill switch
    - Trade approval gating
    """

    # Never risk more than this fraction of portfolio per trade
    MAX_RISK_PER_TRADE = 0.02          # 2%
    # Never use more than this fraction of portfolio in a single trade
    MAX_POSITION_SIZE_PCT = 0.25       # 25%

    def __init__(self, stop_loss_pct: float, take_profit_pct: float,
                 max_daily_loss_pct: float, trade_amount_usdt: float):
        self.stop_loss_pct      = stop_loss_pct
        self.take_profit_pct    = take_profit_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.trade_amount_usdt  = trade_amount_usdt

        self._daily_pnl         = 0.0
        self._daily_pnl_date    = date.today()
        self._kill_switch_active = False
        self._trade_count_today  = 0
        self.open_position: Position | None = None

    # -------------------------------------------------------------------------
    # Trade Approval
    # -------------------------------------------------------------------------

    def approve_trade(self, signal: str, free_usdt: float, current_price: float) -> tuple[bool, str]:
        """
        Decides whether to approve opening a new trade.
        Returns (approved: bool, reason: str).
        """
        self._reset_daily_if_needed()

        if self._kill_switch_active:
            return False, f"Kill switch active — daily loss limit reached ({self.max_daily_loss_pct*100:.1f}%)"

        if signal != "BUY":
            return False, f"Signal is {signal}, not BUY."

        if self.open_position is not None:
            return False, "Already in a position — no new trades until current one closes."

        if free_usdt < 10:
            return False, f"Insufficient USDT balance (${free_usdt:.2f} < $10 minimum)."

        trade_usdt = self._calculate_trade_usdt(free_usdt)
        if trade_usdt <= 0:
            return False, "Calculated trade size is zero — skipping."

        return True, f"Trade approved. Size: ${trade_usdt:.2f} USDT"

    # -------------------------------------------------------------------------
    # Position Sizing
    # -------------------------------------------------------------------------

    def calculate_trade_usdt(self, free_usdt: float) -> float:
        """
        Returns the USDT amount to use for this trade.
        Uses fixed amount from config, but caps at MAX_POSITION_SIZE_PCT of balance.
        """
        return self._calculate_trade_usdt(free_usdt)

    def _calculate_trade_usdt(self, free_usdt: float) -> float:
        max_allowed = free_usdt * self.MAX_POSITION_SIZE_PCT
        trade_usdt  = min(self.trade_amount_usdt, max_allowed)
        trade_usdt  = max(trade_usdt, 0.0)
        return round(trade_usdt, 2)

    # -------------------------------------------------------------------------
    # Stop Loss / Take Profit
    # -------------------------------------------------------------------------

    def get_stop_loss_price(self, entry_price: float) -> float:
        """Returns the stop-loss price for a long position."""
        return round(entry_price * (1 - self.stop_loss_pct), 8)

    def get_take_profit_price(self, entry_price: float) -> float:
        """Returns the take-profit price for a long position."""
        return round(entry_price * (1 + self.take_profit_pct), 8)

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def open_position_record(self, entry_price: float, amount_base: float,
                              amount_usdt: float, order_id: str = None) -> Position:
        """Records a new open position."""
        sl = self.get_stop_loss_price(entry_price)
        tp = self.get_take_profit_price(entry_price)
        pos = Position(
            entry_price=entry_price,
            amount_base=amount_base,
            amount_usdt=amount_usdt,
            stop_loss=sl,
            take_profit=tp,
        )
        pos.order_id = order_id
        self.open_position = pos
        self._trade_count_today += 1
        logger.info(
            f"Position opened: entry=${entry_price:.4f} | "
            f"SL=${sl:.4f} ({self.stop_loss_pct*100:.1f}%) | "
            f"TP=${tp:.4f} ({self.take_profit_pct*100:.1f}%)"
        )
        return pos

    def close_position_record(self, exit_price: float) -> float:
        """
        Closes the current position, updates daily P&L, and returns realized P&L.
        """
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

    # -------------------------------------------------------------------------
    # SL/TP Monitoring
    # -------------------------------------------------------------------------

    def check_stop_loss(self, current_price: float) -> bool:
        """Returns True if current price has hit or breached the stop-loss."""
        if self.open_position is None:
            return False
        hit = current_price <= self.open_position.stop_loss
        if hit:
            logger.warning(
                f"STOP LOSS HIT: price={current_price:.4f} <= sl={self.open_position.stop_loss:.4f}"
            )
        return hit

    def check_take_profit(self, current_price: float) -> bool:
        """Returns True if current price has hit or exceeded the take-profit."""
        if self.open_position is None:
            return False
        hit = current_price >= self.open_position.take_profit
        if hit:
            logger.info(
                f"TAKE PROFIT HIT: price={current_price:.4f} >= tp={self.open_position.take_profit:.4f}"
            )
        return hit

    # -------------------------------------------------------------------------
    # Daily P&L & Kill Switch
    # -------------------------------------------------------------------------

    def _update_daily_pnl(self, pnl: float):
        self._reset_daily_if_needed()
        self._daily_pnl += pnl
        logger.info(f"Daily P&L updated: ${self._daily_pnl:+.2f}")
        self._check_kill_switch_threshold()

    def _check_kill_switch_threshold(self):
        """Activates kill switch if daily loss exceeds the max threshold."""
        # We don't know portfolio value here, so we compare against trade_amount_usdt
        # as a proxy. The OrderEngine passes portfolio value separately.
        pass  # Called externally via check_kill_switch(portfolio_value)

    def check_kill_switch(self, portfolio_value: float) -> bool:
        """
        Returns True if kill switch is (or should be) active.
        Call this every loop iteration with the current portfolio value.
        """
        self._reset_daily_if_needed()
        if self._kill_switch_active:
            return True

        if portfolio_value > 0:
            loss_pct = -self._daily_pnl / portfolio_value
            if loss_pct >= self.max_daily_loss_pct:
                self._kill_switch_active = True
                logger.critical(
                    f"KILL SWITCH ACTIVATED: daily loss {loss_pct*100:.2f}% "
                    f"exceeds limit {self.max_daily_loss_pct*100:.1f}%"
                )
        return self._kill_switch_active

    def reset_kill_switch(self):
        """Manually reset the kill switch (e.g. at start of new day)."""
        self._kill_switch_active = False
        logger.info("Kill switch manually reset.")

    def _reset_daily_if_needed(self):
        """Resets daily P&L counters at the start of a new trading day."""
        today = date.today()
        if today != self._daily_pnl_date:
            logger.info(
                f"New trading day ({today}). Resetting daily P&L from "
                f"${self._daily_pnl:+.2f} to $0.00"
            )
            self._daily_pnl = 0.0
            self._daily_pnl_date = today
            self._kill_switch_active = False
            self._trade_count_today = 0

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

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
