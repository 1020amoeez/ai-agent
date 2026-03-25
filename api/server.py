"""
api/server.py - FastAPI web dashboard for the AI Trading Agent.

Runs the agent in a background thread and exposes:
  REST  → /api/status, /api/start, /api/stop, /api/history, /api/config
  WS    → /ws  (live push every 3 seconds)
  UI    → /    (serves the HTML dashboard)
"""

import sys
import os
import asyncio
import threading
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to path so imports work when run from api/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("api.server")

# ── Agent state (shared between API and background thread) ────────────────────

class AgentState:
    def __init__(self):
        self.running          = False
        self.thread: threading.Thread | None = None
        self.stop_event       = threading.Event()

        # Live data — updated every loop tick
        self.last_signal      = {"signal": "HOLD", "reason": "Agent not started", "confidence": 0, "price": 0, "rsi": 0, "macd_hist": 0, "indicators": {}}
        self.market_summary   = {}
        self.position_status  = {"in_position": False}
        self.balance_usdt     = 0.0
        self.daily_pnl        = 0.0
        self.trade_history    = []
        self.risk_stats       = {}
        self.error_log        = []
        self.iteration        = 0
        self.last_updated     = None
        self.paper_balance    = 1000.0

        # Config snapshot (set on start)
        self.config_snapshot  = {}

    def to_dict(self):
        return {
            "running":         self.running,
            "iteration":       self.iteration,
            "last_updated":    self.last_updated,
            "signal":          self.last_signal,
            "market":          self.market_summary,
            "position":        self.position_status,
            "balance_usdt":    round(self.balance_usdt, 4),
            "daily_pnl":       round(self.daily_pnl, 4),
            "trade_history":   self.trade_history[-50:],   # last 50 trades
            "risk_stats":      self.risk_stats,
            "error_log":       self.error_log[-20:],
            "config":          self.config_snapshot,
            "paper_balance":   round(self.paper_balance, 4),
        }


agent_state = AgentState()


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── Background agent thread ───────────────────────────────────────────────────

def _run_agent_thread(stop_event: threading.Event):
    """Runs the trading agent loop in a background thread."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    try:
        from config import Config, ConfigError
        cfg = Config()
    except Exception as e:
        agent_state.error_log.append({"time": str(datetime.utcnow()), "error": f"Config error: {e}"})
        agent_state.running = False
        return

    agent_state.config_snapshot = {
        "exchange":          cfg.exchange,
        "symbol":            cfg.symbol,
        "timeframe":         cfg.timeframe,
        "mode":              cfg.trading_mode,
        "trade_amount_usdt": cfg.trade_amount_usdt,
        "stop_loss_pct":     cfg.stop_loss_pct,
        "take_profit_pct":   cfg.take_profit_pct,
        "loop_interval_s":   cfg.loop_interval_seconds,
    }

    try:
        from exchange.connector import ExchangeConnector
        from data.market_data import MarketData
        from strategy.signal_engine import SignalEngine
        from risk.risk_manager import RiskManager
        from execution.order_engine import OrderEngine
        from notifications.telegram_bot import TelegramNotifier

        notifier  = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        connector = ExchangeConnector(cfg.exchange, cfg.api_key, cfg.api_secret, cfg.is_paper)

        if cfg.is_live:
            connector.load_markets(cfg.symbol)

        market_data    = MarketData(connector, cfg.symbol, cfg.timeframe, cfg.candle_limit)
        signal_engine  = SignalEngine()
        risk_manager   = RiskManager(cfg.stop_loss_pct, cfg.take_profit_pct,
                                     cfg.max_daily_loss_pct, cfg.trade_amount_usdt)
        order_engine   = OrderEngine(connector, risk_manager, notifier,
                                     cfg.symbol, cfg.is_paper)

        notifier.send_startup(cfg.symbol, cfg.exchange, cfg.timeframe,
                              cfg.trading_mode, cfg.trade_amount_usdt)

    except Exception as e:
        agent_state.error_log.append({"time": str(datetime.utcnow()), "error": f"Init error: {e}"})
        agent_state.running = False
        return

    logger.info("Agent background thread started.")

    while not stop_event.is_set():
        agent_state.iteration += 1
        try:
            df      = market_data.fetch()
            summary = market_data.summary(df)
            agent_state.market_summary = summary

            try:
                current_price = connector.get_current_price(cfg.symbol)
            except Exception:
                current_price = summary["close"]

            # Monitor SL/TP
            if order_engine.is_in_position:
                order_engine.monitor_position(current_price)

            # Kill switch
            balance = connector.get_free_usdt()
            portfolio_value = balance + (
                (risk_manager.open_position.amount_base * current_price)
                if risk_manager.is_in_position else 0
            )
            risk_manager.check_kill_switch(portfolio_value)

            # Generate signal
            signal = signal_engine.evaluate(df, in_position=order_engine.is_in_position)
            agent_state.last_signal = signal

            # Act on signal
            if signal["signal"] == "BUY" and not order_engine.is_in_position:
                notifier.send_signal(signal)
                order_engine.execute_buy(signal)

            elif signal["signal"] == "SELL" and order_engine.is_in_position:
                notifier.send_signal(signal)
                order_engine.execute_sell(signal["reason"], current_price)

            # Update shared state
            agent_state.balance_usdt    = balance
            agent_state.daily_pnl       = risk_manager.daily_pnl
            agent_state.position_status = order_engine.get_position_status(current_price)
            agent_state.trade_history   = order_engine.get_trade_history()
            agent_state.risk_stats      = risk_manager.get_stats()
            agent_state.last_updated    = datetime.utcnow().isoformat()

            if cfg.is_paper:
                agent_state.paper_balance = connector.get_paper_balance()

        except Exception as e:
            err = {"time": str(datetime.utcnow()), "error": str(e)}
            agent_state.error_log.append(err)
            logger.exception(f"Agent loop error: {e}")

        # Sleep in small increments so stop_event is responsive
        for _ in range(cfg.loop_interval_seconds):
            if stop_event.is_set():
                break
            time.sleep(1)

    agent_state.running = False
    logger.info("Agent background thread stopped.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background WS broadcaster
    asyncio.create_task(_ws_broadcaster())
    yield

app = FastAPI(title="AI Trading Agent", version="1.0.0", lifespan=lifespan)

# Serve static files (index.html)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/status")
async def get_status():
    return JSONResponse(agent_state.to_dict())


@app.post("/api/start")
async def start_agent():
    if agent_state.running:
        raise HTTPException(status_code=400, detail="Agent is already running.")

    agent_state.stop_event = threading.Event()
    agent_state.running    = True
    agent_state.error_log  = []

    t = threading.Thread(
        target=_run_agent_thread,
        args=(agent_state.stop_event,),
        daemon=True,
        name="trading-agent",
    )
    t.start()
    agent_state.thread = t

    return {"status": "started", "message": "Agent started successfully."}


@app.post("/api/stop")
async def stop_agent():
    if not agent_state.running:
        raise HTTPException(status_code=400, detail="Agent is not running.")

    agent_state.stop_event.set()
    agent_state.running = False

    return {"status": "stopped", "message": "Agent stop signal sent."}


@app.get("/api/history")
async def get_history():
    return JSONResponse({"trades": agent_state.trade_history})


@app.get("/api/config")
async def get_config():
    return JSONResponse(agent_state.config_snapshot)


@app.get("/api/errors")
async def get_errors():
    return JSONResponse({"errors": agent_state.error_log})


@app.get("/health")
async def health():
    return {"status": "ok", "agent_running": agent_state.running}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current state immediately on connect
        await ws.send_json(agent_state.to_dict())
        while True:
            # Keep connection alive — data is pushed by broadcaster
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def _ws_broadcaster():
    """Pushes live state to all connected WebSocket clients every 3 seconds."""
    while True:
        await asyncio.sleep(3)
        if manager.active:
            await manager.broadcast(agent_state.to_dict())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)
