"""
execution/order_engine.py - Executes and monitors trades automatically.
Coordinates between the exchange connector, risk manager, and notifier.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderEngine:
    """
    Handles the full lifecycle of a trade:
        open → monitor (SL/TP) → close
    No manual intervention required.
    """

    def __init__(self, connector, risk_manager, notifier, symbol: str, paper_mode: bool = True):
        self.connector    = connector
        self.risk         = risk_manager
        self.notifier     = notifier
        self.symbol       = symbol
        self.paper_mode   = paper_mode
        self._trade_log   = []          # in-memory trade history

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def is_in_position(self) -> bool:
        return self.risk.is_in_position

    def execute_buy(self, signal: dict) -> bool:
        """
        Executes a BUY order.
        1. Checks balance and risk approval
        2. Places market buy
        3. Records position in risk manager
        4. Sends notification

        Returns True if order was placed, False otherwise.
        """
        free_usdt    = self.connector.get_free_usdt()
        current_price = signal["price"]

        approved, reason = self.risk.approve_trade("BUY", free_usdt, current_price)
        if not approved:
            logger.info(f"Trade not approved: {reason}")
            return False

        trade_usdt = self.risk.calculate_trade_usdt(free_usdt)
        logger.info(
            f"Executing BUY | Symbol: {self.symbol} | "
            f"Amount: ${trade_usdt:.2f} USDT | Price: ${current_price:.4f}"
        )

        try:
            order = self.connector.place_market_buy(self.symbol, trade_usdt)
        except Exception as e:
            logger.error(f"BUY order failed: {e}")
            self.notifier.send_error(f"BUY order FAILED for {self.symbol}: {e}")
            return False

        entry_price  = float(order.get("price") or current_price)
        amount_base  = float(order.get("amount") or (trade_usdt / entry_price))
        amount_spent = float(order.get("cost") or trade_usdt)

        position = self.risk.open_position_record(
            entry_price=entry_price,
            amount_base=amount_base,
            amount_usdt=amount_spent,
            order_id=order.get("id"),
        )

        self._log_trade("BUY", entry_price, amount_base, amount_spent, signal.get("reason", ""))

        self.notifier.send_trade_executed(
            side="BUY",
            symbol=self.symbol,
            price=entry_price,
            amount=amount_base,
            trade_usdt=amount_spent,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            reason=signal.get("reason", ""),
            paper=self.paper_mode,
        )
        return True

    def execute_sell(self, reason: str, current_price: float) -> bool:
        """
        Executes a SELL order to close the current position.
        1. Gets position info from risk manager
        2. Places market sell
        3. Calculates realized P&L
        4. Sends notification

        Returns True if order was placed, False otherwise.
        """
        if not self.is_in_position:
            logger.warning("execute_sell called but no open position found.")
            return False

        position     = self.risk.open_position
        amount_base  = position.amount_base

        logger.info(
            f"Executing SELL | Symbol: {self.symbol} | "
            f"Amount: {amount_base:.6f} | Price: ${current_price:.4f} | Reason: {reason}"
        )

        try:
            order = self.connector.place_market_sell(self.symbol, amount_base)
        except Exception as e:
            logger.error(f"SELL order failed: {e}")
            self.notifier.send_error(f"SELL order FAILED for {self.symbol}: {e}")
            return False

        exit_price  = float(order.get("price") or current_price)
        pnl         = self.risk.close_position_record(exit_price)
        pnl_pct     = (pnl / position.amount_usdt * 100) if position.amount_usdt > 0 else 0

        self._log_trade("SELL", exit_price, amount_base, exit_price * amount_base, reason, pnl)

        self.notifier.send_trade_executed(
            side="SELL",
            symbol=self.symbol,
            price=exit_price,
            amount=amount_base,
            trade_usdt=exit_price * amount_base,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            paper=self.paper_mode,
        )
        return True

    def monitor_position(self, current_price: float) -> str | None:
        """
        Checks if the open position has hit stop-loss or take-profit.
        Executes sell automatically if either is triggered.

        Returns the exit reason string, or None if still holding.
        """
        if not self.is_in_position:
            return None

        position = self.risk.open_position
        pnl      = position.unrealized_pnl(current_price)
        pnl_pct  = position.pnl_pct(current_price)

        logger.debug(
            f"Position monitor | Price: ${current_price:.4f} | "
            f"SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f} | "
            f"Unrealized P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
        )

        if self.risk.check_stop_loss(current_price):
            reason = f"Stop-loss triggered at ${current_price:.4f} (SL=${position.stop_loss:.4f})"
            self.execute_sell(reason, current_price)
            return reason

        if self.risk.check_take_profit(current_price):
            reason = f"Take-profit triggered at ${current_price:.4f} (TP=${position.take_profit:.4f})"
            self.execute_sell(reason, current_price)
            return reason

        return None

    def get_position_status(self, current_price: float) -> dict:
        """Returns a summary of the current position state."""
        if not self.is_in_position:
            return {"in_position": False}

        pos = self.risk.open_position
        pnl = pos.unrealized_pnl(current_price)
        return {
            "in_position":  True,
            "symbol":       self.symbol,
            "entry_price":  pos.entry_price,
            "current_price": current_price,
            "amount_base":  pos.amount_base,
            "amount_usdt":  pos.amount_usdt,
            "stop_loss":    pos.stop_loss,
            "take_profit":  pos.take_profit,
            "unrealized_pnl": round(pnl, 4),
            "pnl_pct":      round(pos.pnl_pct(current_price), 2),
            "opened_at":    str(pos.opened_at),
        }

    def get_trade_history(self) -> list:
        """Returns a copy of the in-memory trade log."""
        return list(self._trade_log)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _log_trade(self, side: str, price: float, amount_base: float,
                   amount_usdt: float, reason: str, pnl: float = None):
        record = {
            "timestamp":   datetime.utcnow().isoformat(),
            "side":        side,
            "symbol":      self.symbol,
            "price":       round(price, 4),
            "amount_base": round(amount_base, 8),
            "amount_usdt": round(amount_usdt, 4),
            "reason":      reason,
            "paper":       self.paper_mode,
        }
        if pnl is not None:
            record["pnl"] = round(pnl, 4)
        self._trade_log.append(record)
        logger.info(f"Trade logged: {record}")
