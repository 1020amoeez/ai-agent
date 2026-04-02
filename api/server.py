"""
api/server.py - Unified dashboard for Crypto + Forex trading agents.
Credentials are passed via API at start time — no .env needed.
Multiple users can run independent sessions.
"""

import sys, os, asyncio, threading, time, logging, importlib.util
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

CRYPTO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CRYPTO_ROOT)

FOREX_PATH = os.path.abspath(os.path.join(CRYPTO_ROOT, "..", "Ai forex agent"))
# NOTE: do NOT add FOREX_PATH to sys.path — forex modules are loaded via importlib
# to avoid clashing with crypto modules that share the same package names (data, strategy, etc.)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("api.server")

# ── Request models ────────────────────────────────────────────────────────────

class CryptoStartRequest(BaseModel):
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    trade_amount_usdt: float = 50.0
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    trading_mode: str = "paper"
    telegram_token: str = ""
    telegram_chat_id: str = ""

class ForexStartRequest(BaseModel):
    mt5_login: str = ""
    mt5_password: str = ""
    mt5_server: str = "Exness-MT5Real"
    symbol: str = "EURUSDm"
    timeframe: str = "M5"
    lot_size: float = 0.01
    stop_loss_pips: int = 30
    take_profit_pips: int = 60
    max_daily_loss: float = 20.0
    max_spread_points: int = 20
    trading_mode: str = "paper"
    starting_balance: float = 10000.0
    telegram_token: str = ""
    telegram_chat_id: str = ""

# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState:
    def __init__(self, agent_type: str):
        self.agent_type      = agent_type
        self.running         = False
        self.thread          = None
        self.stop_event      = threading.Event()
        self.last_signal     = {"signal": "HOLD", "reason": "Agent not started", "confidence": 0, "price": 0, "rsi": 0, "macd_hist": 0, "indicators": {}}
        self.market_summary  = {}
        self.position_status = {"in_position": False}
        self.account_info    = {}
        self.balance_usdt    = 0.0
        self.paper_balance   = 1000.0
        self.daily_pnl       = 0.0
        self.trade_history   = []
        self.pnl_history     = []  # [{time, pnl, cumulative, balance, type}]
        self.total_pnl       = 0.0
        self.win_count       = 0
        self.loss_count      = 0
        self.risk_stats      = {}
        self.error_log       = []
        self.iteration       = 0
        self.last_updated    = None
        self.config_snapshot = {}

    def record_pnl(self, pnl: float, balance: float, reason: str = ""):
        """Record a P&L event for history tracking."""
        self.total_pnl += pnl
        if pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.pnl_history.append({
            "time": datetime.utcnow().isoformat(),
            "pnl": round(pnl, 4),
            "cumulative": round(self.total_pnl, 4),
            "balance": round(balance, 4),
            "type": "WIN" if pnl >= 0 else "LOSS",
            "reason": reason[:60],
        })

    def to_dict(self):
        import math
        def _clean(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            return obj
        total_trades = self.win_count + self.loss_count
        return _clean({
            "agent_type":    self.agent_type,
            "running":       self.running,
            "iteration":     self.iteration,
            "last_updated":  self.last_updated,
            "signal":        self.last_signal,
            "market":        self.market_summary,
            "position":      self.position_status,
            "account":       self.account_info,
            "balance_usdt":  round(self.balance_usdt, 4),
            "paper_balance": round(self.paper_balance, 4),
            "daily_pnl":     round(self.daily_pnl, 2),
            "total_pnl":     round(self.total_pnl, 4),
            "win_count":     self.win_count,
            "loss_count":    self.loss_count,
            "win_rate":      round((self.win_count / total_trades * 100) if total_trades > 0 else 0, 1),
            "trade_history": self.trade_history[-50:],
            "pnl_history":   self.pnl_history[-100:],
            "risk_stats":    self.risk_stats,
            "error_log":     self.error_log[-20:],
            "config":        self.config_snapshot,
        })

crypto_state = AgentState("crypto")
forex_state  = AgentState("forex")

# ── WebSocket manager ─────────────────────────────────────────────────────────

class WSManager:
    def __init__(self):
        self.active = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.disconnect(d)

crypto_ws = WSManager()
forex_ws  = WSManager()

# ── Crypto agent thread ───────────────────────────────────────────────────────

def _run_crypto(stop_event, req: CryptoStartRequest):
    # Validate timeframe
    TIMEFRAME_SECONDS = {
        "1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,
        "1h":3600,"2h":7200,"4h":14400,"6h":21600,"8h":28800,
        "12h":43200,"1d":86400
    }
    loop_secs = 2  # refresh every 2 seconds

    crypto_state.config_snapshot = {
        "type": "crypto",
        "exchange": req.exchange,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "mode": req.trading_mode,
        "trade_amount_usdt": req.trade_amount_usdt,
        "risk_per_trade_pct": req.risk_per_trade_pct,
        "max_daily_loss_pct": req.max_daily_loss_pct,
        "sl_tp": "Dynamic (ATR-based)",
        "loop_interval_s": loop_secs,
    }

    try:
        from exchange.connector import ExchangeConnector
        from data.market_data import MarketData
        from strategy.signal_engine import SignalEngine
        from risk.risk_manager import RiskManager
        from execution.order_engine import OrderEngine
        from notifications.telegram_bot import TelegramNotifier

        is_paper  = req.trading_mode == "paper"
        notifier  = TelegramNotifier(req.telegram_token, req.telegram_chat_id)
        connector = ExchangeConnector(req.exchange, req.api_key, req.api_secret, is_paper)
        if not is_paper:
            connector.load_markets(req.symbol)
        market  = MarketData(connector, req.symbol, req.timeframe, 200)
        engine  = SignalEngine()
        risk    = RiskManager(req.max_daily_loss_pct, req.trade_amount_usdt,
                              req.risk_per_trade_pct)
        orders  = OrderEngine(connector, risk, notifier, req.symbol, is_paper)
        notifier.send_startup(req.symbol, req.exchange, req.timeframe,
                              req.trading_mode, req.trade_amount_usdt)
    except Exception as e:
        crypto_state.error_log.append({"time": str(datetime.utcnow()), "error": f"Init failed: {e}"})
        crypto_state.running = False
        return

    prev_trade_count = 0

    while not stop_event.is_set():
        crypto_state.iteration += 1
        try:
            df      = market.fetch()
            summary = market.summary(df)
            crypto_state.market_summary = summary

            try:
                current_price = connector.get_current_price(req.symbol)
            except Exception:
                current_price = summary["close"]

            if orders.is_in_position:
                orders.monitor_position(current_price)

            balance = connector.get_free_usdt()
            portfolio_value = balance + (
                (risk.open_position.amount_base * current_price) if risk.is_in_position else 0
            )
            risk.check_kill_switch(portfolio_value)

            if not risk.kill_switch_active:
                signal = engine.evaluate(df, in_position=orders.is_in_position)
                crypto_state.last_signal = signal
                if signal["signal"] == "BUY" and not orders.is_in_position:
                    notifier.send_signal(signal)
                    orders.execute_buy(signal)
                elif signal["signal"] == "SELL" and orders.is_in_position:
                    notifier.send_signal(signal)
                    orders.execute_sell(signal["reason"], current_price)
            else:
                crypto_state.last_signal = {"signal": "HOLD", "reason": "Kill switch active",
                    "confidence": 0, "price": current_price,
                    "rsi": summary.get("rsi", 0), "macd_hist": summary.get("macd_hist", 0), "indicators": {}}

            # Track PnL history - detect new SELL trades (closed positions)
            current_trades = orders.get_trade_history()
            if len(current_trades) > prev_trade_count:
                for t in current_trades[prev_trade_count:]:
                    if t.get("side") == "SELL" and "pnl" in t:
                        bal = connector.get_paper_balance() if is_paper else connector.get_free_usdt()
                        crypto_state.record_pnl(t["pnl"], bal, t.get("reason", ""))
                prev_trade_count = len(current_trades)

            crypto_state.balance_usdt    = balance
            crypto_state.paper_balance   = connector.get_paper_balance() if is_paper else balance
            crypto_state.daily_pnl       = risk.daily_pnl
            crypto_state.position_status = orders.get_position_status(current_price)
            crypto_state.trade_history   = current_trades
            crypto_state.risk_stats      = risk.get_stats()
            crypto_state.last_updated    = datetime.utcnow().isoformat()

        except Exception as e:
            crypto_state.error_log.append({"time": str(datetime.utcnow()), "error": str(e)})
            logger.exception(f"Crypto loop error: {e}")

        for _ in range(loop_secs):
            if stop_event.is_set(): break
            time.sleep(1)

    crypto_state.running = False

# ── Forex agent thread ────────────────────────────────────────────────────────

def _load_forex_module(name, rel_path):
    path = os.path.join(FOREX_PATH, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TIMEFRAME_MAP = {
    "M1":1,"M2":2,"M3":3,"M4":4,"M5":5,"M6":6,"M10":10,"M12":12,
    "M15":15,"M20":20,"M30":30,"H1":16385,"H2":16386,"H3":16387,
    "H4":16388,"H6":16390,"H8":16392,"H12":16396,"D1":16408,
}
TIMEFRAME_SECS_FOREX = {
    "M1":60,"M5":300,"M15":900,"M30":1800,
    "H1":3600,"H4":14400,"D1":86400,
}

def _run_forex(stop_event, req: ForexStartRequest):
    loop_secs = 2  # refresh every 2 seconds
    tf_const  = TIMEFRAME_MAP.get(req.timeframe, 5)

    forex_state.config_snapshot = {
        "type": "forex",
        "server": req.mt5_server,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "mode": req.trading_mode,
        "lot_size": req.lot_size,
        "stop_loss_pips": req.stop_loss_pips,
        "take_profit_pips": req.take_profit_pips,
        "loop_interval_s": loop_secs,
    }

    try:
        mt5_mod   = _load_forex_module("mt5_conn",    "mt5/connector.py")
        data_mod  = _load_forex_module("forex_data",  "data/market_data.py")
        sig_mod   = _load_forex_module("forex_sig",   "strategy/signal_engine.py")
        risk_mod  = _load_forex_module("forex_risk",  "risk/risk_manager.py")
        ord_mod   = _load_forex_module("forex_ord",   "execution/order_engine.py")
        notif_mod = _load_forex_module("forex_notif", "notifications/telegram_bot.py")

        is_paper  = req.trading_mode == "paper"
        notifier  = notif_mod.TelegramNotifier(req.telegram_token, req.telegram_chat_id)
        connector = mt5_mod.MT5Connector(req.mt5_login, req.mt5_password, req.mt5_server, is_paper, req.starting_balance)
        market    = data_mod.ForexMarketData(connector, req.symbol, tf_const, 200)
        engine    = sig_mod.ForexSignalEngine()
        risk      = risk_mod.ForexRiskManager(req.stop_loss_pips, req.take_profit_pips,
                                              req.max_daily_loss, req.lot_size,
                                              req.max_spread_points)
        orders    = ord_mod.ForexOrderEngine(connector, risk, notifier, req.symbol, is_paper)
        notifier.send_startup(req.symbol, req.mt5_server, req.timeframe,
                              req.trading_mode, req.lot_size)
    except Exception as e:
        forex_state.error_log.append({"time": str(datetime.utcnow()), "error": f"Init failed: {e}"})
        forex_state.running = False
        return

    while not stop_event.is_set():
        forex_state.iteration += 1
        try:
            df      = market.fetch()
            summary = market.summary(df)
            forex_state.market_summary = summary

            tick          = connector.get_tick(req.symbol)
            current_price = tick.get("last") or summary["close"]
            forex_state.market_summary["price"] = current_price

            if orders.is_in_position:
                orders.monitor_position(current_price)

            acc     = connector.get_account_info()
            balance = acc.get("balance", 0)
            forex_state.account_info = acc

            risk.check_kill_switch(balance)

            if not risk.kill_switch_active:
                signal = engine.evaluate(df, in_position=orders.is_in_position,
                                         position_type=orders.position_type)
                forex_state.last_signal = signal
                if signal["signal"] in ("BUY","SELL") and not orders.is_in_position:
                    notifier.send_signal(signal)
                    orders.execute_trade(signal)
                elif signal["signal"] in ("BUY","SELL") and orders.is_in_position:
                    notifier.send_signal(signal)
                    orders.close_trade(signal["reason"])
            else:
                forex_state.last_signal = {"signal": "HOLD", "reason": "Kill switch active",
                    "confidence": 0, "price": current_price,
                    "rsi": summary.get("rsi", 0), "macd_hist": summary.get("macd_hist", 0), "indicators": {}}

            forex_state.balance_usdt    = balance
            forex_state.daily_pnl       = risk.daily_pnl
            forex_state.position_status = orders.get_position_status(current_price)
            forex_state.trade_history   = orders.get_trade_history()
            forex_state.risk_stats      = risk.get_stats()
            forex_state.last_updated    = datetime.utcnow().isoformat()

        except Exception as e:
            forex_state.error_log.append({"time": str(datetime.utcnow()), "error": str(e)})
            logger.exception(f"Forex loop error: {e}")

        for _ in range(loop_secs):
            if stop_event.is_set(): break
            time.sleep(1)

    forex_state.running = False

# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_broadcaster(crypto_ws, crypto_state))
    asyncio.create_task(_broadcaster(forex_ws,  forex_state))
    yield

app = FastAPI(title="AI Trading Dashboard", lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ── Crypto endpoints ──────────────────────────────────────────────────────────

@app.get("/api/crypto/status")
async def crypto_status():
    return JSONResponse(crypto_state.to_dict())

@app.post("/api/crypto/start")
async def crypto_start(req: CryptoStartRequest):
    if crypto_state.running:
        raise HTTPException(400, "Crypto agent already running")
    if req.trading_mode == "live" and (not req.api_key or not req.api_secret):
        raise HTTPException(400, "API Key and Secret are required for live trading")
    crypto_state.stop_event = threading.Event()
    crypto_state.running    = True
    crypto_state.error_log  = []
    crypto_state.iteration  = 0
    t = threading.Thread(target=_run_crypto, args=(crypto_state.stop_event, req), daemon=True)
    t.start()
    crypto_state.thread = t
    return {"status": "started", "agent": "crypto"}

@app.post("/api/crypto/stop")
async def crypto_stop():
    if not crypto_state.running:
        raise HTTPException(400, "Crypto agent not running")
    crypto_state.stop_event.set()
    crypto_state.running = False
    return {"status": "stopped", "agent": "crypto"}

# ── Forex endpoints ───────────────────────────────────────────────────────────

@app.get("/api/forex/status")
async def forex_status():
    return JSONResponse(forex_state.to_dict())

@app.post("/api/forex/start")
async def forex_start(req: ForexStartRequest):
    if forex_state.running:
        raise HTTPException(400, "Forex agent already running")
    if req.trading_mode == "live" and (not req.mt5_login or not req.mt5_password):
        raise HTTPException(400, "MT5 Login and Password are required for live trading")
    forex_state.stop_event = threading.Event()
    forex_state.running    = True
    forex_state.error_log  = []
    forex_state.iteration  = 0
    t = threading.Thread(target=_run_forex, args=(forex_state.stop_event, req), daemon=True)
    t.start()
    forex_state.thread = t
    return {"status": "started", "agent": "forex"}

@app.post("/api/forex/stop")
async def forex_stop():
    if not forex_state.running:
        raise HTTPException(400, "Forex agent not running")
    forex_state.stop_event.set()
    forex_state.running = False
    return {"status": "stopped", "agent": "forex"}

# ── WebSockets ────────────────────────────────────────────────────────────────

@app.websocket("/ws/crypto")
async def ws_crypto(ws: WebSocket):
    await crypto_ws.connect(ws)
    try:
        await ws.send_json(crypto_state.to_dict())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        crypto_ws.disconnect(ws)

@app.websocket("/ws/forex")
async def ws_forex(ws: WebSocket):
    await forex_ws.connect(ws)
    try:
        await ws.send_json(forex_state.to_dict())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        forex_ws.disconnect(ws)

async def _broadcaster(manager: WSManager, state: AgentState):
    while True:
        await asyncio.sleep(2)
        if manager.active:
            await manager.broadcast(state.to_dict())

@app.get("/health")
async def health():
    return {"status": "ok", "crypto_running": crypto_state.running, "forex_running": forex_state.running}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)
