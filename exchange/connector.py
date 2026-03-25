"""
exchange/connector.py - Unified exchange connector using ccxt.
Supports any ccxt-compatible exchange. Handles both paper and live trading.
"""

import time
import logging
import ccxt

logger = logging.getLogger(__name__)


class PaperPosition:
    """Tracks a simulated open position for paper trading."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.side = None
        self.entry_price = None
        self.amount_base = 0.0
        self.amount_usdt = 0.0
        self.order_id = None

    @property
    def is_open(self):
        return self.side is not None


class ExchangeConnector:
    """
    Wraps ccxt to provide a clean interface for the trading agent.
    Works in both 'paper' (simulated) and 'live' (real) modes.
    """

    def __init__(self, exchange_id: str, api_key: str, api_secret: str, paper_mode: bool = True):
        self.exchange_id = exchange_id
        self.paper_mode = paper_mode
        self._paper_position = PaperPosition()
        self._paper_balance_usdt = 1000.0   # simulated starting balance
        self._paper_order_counter = 1

        # Build ccxt exchange instance
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        self._exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if paper_mode:
            logger.info(f"[PAPER] Exchange connector initialized for '{exchange_id}' in paper trading mode.")
        else:
            logger.info(f"[LIVE] Exchange connector initialized for '{exchange_id}' in LIVE trading mode.")

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list:
        """
        Fetch OHLCV candles.
        Returns list of [timestamp, open, high, low, close, volume].
        """
        try:
            candles = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            logger.debug(f"Fetched {len(candles)} candles for {symbol} @ {timeframe}")
            return candles
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching OHLCV: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching OHLCV: {e}")
            raise

    def fetch_ticker(self, symbol: str) -> dict:
        """
        Fetch latest ticker for a symbol.
        Returns dict with 'last', 'bid', 'ask', 'volume', etc.
        """
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """Returns the current last traded price."""
        ticker = self.fetch_ticker(symbol)
        return float(ticker["last"])

    # -------------------------------------------------------------------------
    # Account / Balance
    # -------------------------------------------------------------------------

    def get_balance(self) -> dict:
        """
        Returns balance dict: {'USDT': float, 'BTC': float, ...}
        In paper mode returns simulated balances.
        """
        if self.paper_mode:
            base = self._paper_position.amount_base if self._paper_position.is_open else 0.0
            quote_currency = "USDT"
            return {
                quote_currency: round(self._paper_balance_usdt, 4),
                "_paper": True,
            }
        try:
            raw = self._exchange.fetch_balance()
            return {k: v["free"] for k, v in raw["total"].items() if v > 0}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    def get_free_usdt(self) -> float:
        """Convenience: returns free USDT (or equivalent quote currency)."""
        if self.paper_mode:
            return self._paper_balance_usdt
        try:
            balance = self._exchange.fetch_balance()
            return float(balance["free"].get("USDT", 0.0))
        except Exception as e:
            logger.error(f"Error fetching free USDT: {e}")
            raise

    def get_base_balance(self, symbol: str) -> float:
        """Returns free balance of the base currency (e.g. BTC in BTC/USDT)."""
        base = symbol.split("/")[0]
        if self.paper_mode:
            return self._paper_position.amount_base if self._paper_position.is_open else 0.0
        try:
            balance = self._exchange.fetch_balance()
            return float(balance["free"].get(base, 0.0))
        except Exception as e:
            logger.error(f"Error fetching base balance for {base}: {e}")
            raise

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def get_open_orders(self, symbol: str) -> list:
        """Returns list of open orders for the symbol."""
        if self.paper_mode:
            return []  # paper mode doesn't track limit orders
        try:
            return self._exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            raise

    def place_market_buy(self, symbol: str, amount_usdt: float) -> dict:
        """
        Places a market BUY order.
        amount_usdt: how many USDT worth to buy.
        Returns order dict with keys: id, price, amount (base), cost (usdt), side, status.
        """
        if self.paper_mode:
            return self._paper_buy(symbol, amount_usdt)
        try:
            # Most exchanges support 'quoteOrderQty' for market buys in USDT
            market = self._exchange.market(symbol)
            # Try cost-based order first
            try:
                order = self._exchange.create_order(
                    symbol, "market", "buy", None,
                    params={"quoteOrderQty": amount_usdt}
                )
            except Exception:
                # Fallback: convert USDT to base amount using ticker
                price = self.get_current_price(symbol)
                base_amount = amount_usdt / price
                base_amount = self._round_amount(symbol, base_amount)
                order = self._exchange.create_market_buy_order(symbol, base_amount)

            logger.info(f"[LIVE] Market BUY executed: {order}")
            return self._normalize_order(order)
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for BUY: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing market BUY: {e}")
            raise

    def place_market_sell(self, symbol: str, amount_base: float) -> dict:
        """
        Places a market SELL order.
        amount_base: amount of base currency to sell (e.g. 0.001 BTC).
        Returns order dict.
        """
        if self.paper_mode:
            return self._paper_sell(symbol, amount_base)
        try:
            amount_base = self._round_amount(symbol, amount_base)
            order = self._exchange.create_market_sell_order(symbol, amount_base)
            logger.info(f"[LIVE] Market SELL executed: {order}")
            return self._normalize_order(order)
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for SELL: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing market SELL: {e}")
            raise

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Fetch a specific order by ID."""
        if self.paper_mode:
            return {"id": order_id, "status": "closed"}
        try:
            order = self._exchange.fetch_order(order_id, symbol)
            return self._normalize_order(order)
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True on success."""
        if self.paper_mode:
            return True
        try:
            self._exchange.cancel_order(order_id, symbol)
            logger.info(f"[LIVE] Order {order_id} cancelled.")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Paper Trading Internals
    # -------------------------------------------------------------------------

    def _paper_buy(self, symbol: str, amount_usdt: float) -> dict:
        """Simulates a market buy in paper mode."""
        price = self.get_current_price(symbol)
        if amount_usdt > self._paper_balance_usdt:
            amount_usdt = self._paper_balance_usdt
        fee = amount_usdt * 0.001       # 0.1% fee
        net_usdt = amount_usdt - fee
        base_amount = net_usdt / price

        self._paper_balance_usdt -= amount_usdt
        self._paper_position.side = "buy"
        self._paper_position.entry_price = price
        self._paper_position.amount_base = base_amount
        self._paper_position.amount_usdt = amount_usdt
        self._paper_position.order_id = f"paper_{self._paper_order_counter}"
        self._paper_order_counter += 1

        order = {
            "id": self._paper_position.order_id,
            "symbol": symbol,
            "side": "buy",
            "price": price,
            "amount": base_amount,
            "cost": amount_usdt,
            "fee": fee,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
        }
        logger.info(f"[PAPER] BUY {base_amount:.6f} {symbol.split('/')[0]} @ ${price:,.2f} | Cost: ${amount_usdt:.2f}")
        return order

    def _paper_sell(self, symbol: str, amount_base: float) -> dict:
        """Simulates a market sell in paper mode."""
        price = self.get_current_price(symbol)
        gross_usdt = amount_base * price
        fee = gross_usdt * 0.001        # 0.1% fee
        net_usdt = gross_usdt - fee

        self._paper_balance_usdt += net_usdt
        entry_price = self._paper_position.entry_price or price
        pnl = net_usdt - self._paper_position.amount_usdt
        self._paper_position.reset()

        order = {
            "id": f"paper_{self._paper_order_counter}",
            "symbol": symbol,
            "side": "sell",
            "price": price,
            "amount": amount_base,
            "cost": gross_usdt,
            "fee": fee,
            "pnl": pnl,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
        }
        self._paper_order_counter += 1
        direction = "PROFIT" if pnl >= 0 else "LOSS"
        logger.info(
            f"[PAPER] SELL {amount_base:.6f} {symbol.split('/')[0]} @ ${price:,.2f} | "
            f"Net: ${net_usdt:.2f} | {direction}: ${pnl:+.2f}"
        )
        return order

    def get_paper_position(self) -> PaperPosition:
        """Returns the current paper trading position."""
        return self._paper_position

    def get_paper_balance(self) -> float:
        """Returns current paper USDT balance."""
        return self._paper_balance_usdt

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _round_amount(self, symbol: str, amount: float) -> float:
        """Round amount to the exchange's precision for a symbol."""
        try:
            return self._exchange.amount_to_precision(symbol, amount)
        except Exception:
            return round(amount, 6)

    def _normalize_order(self, order: dict) -> dict:
        """Normalize ccxt order dict to a consistent internal format."""
        return {
            "id": str(order.get("id", "")),
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "price": float(order.get("average") or order.get("price") or 0),
            "amount": float(order.get("filled") or order.get("amount") or 0),
            "cost": float(order.get("cost") or 0),
            "fee": float((order.get("fee") or {}).get("cost", 0)),
            "status": order.get("status", "unknown"),
            "timestamp": order.get("timestamp", int(time.time() * 1000)),
        }

    def load_markets(self, symbol: str):
        """Pre-load market info (needed before placing orders on some exchanges)."""
        try:
            self._exchange.load_markets()
            if symbol not in self._exchange.markets:
                raise ValueError(f"Symbol '{symbol}' not found on {self.exchange_id}.")
            logger.info(f"Markets loaded. '{symbol}' confirmed available.")
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            raise
