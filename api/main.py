from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    ai_router,
    backtest_router,
    config_router,
    equity_router,
    risk_router,
    status_router,
    strategy_router,
    trades_router,
)
from api.services.container import build_container
from api.services.logging_service import setup_api_logger

logger = setup_api_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = build_container()
    logger.info("api_started", extra={"event": "app.startup"})
    yield
    container = app.state.container
    container.trading_service.stop()
    logger.info("api_stopped", extra={"event": "app.shutdown"})


app = FastAPI(title="Forex Trading Control API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status_router)
app.include_router(risk_router)
app.include_router(trades_router)
app.include_router(equity_router)
app.include_router(strategy_router)
app.include_router(ai_router)
app.include_router(backtest_router)
app.include_router(config_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "forex-control-api"}
