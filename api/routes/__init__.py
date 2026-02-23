from .ai import router as ai_router
from .backtest import router as backtest_router
from .config import router as config_router
from .equity import router as equity_router
from .risk import router as risk_router
from .status import router as status_router
from .strategy import router as strategy_router
from .trades import router as trades_router

__all__ = [
    "ai_router",
    "backtest_router",
    "config_router",
    "equity_router",
    "risk_router",
    "status_router",
    "strategy_router",
    "trades_router",
]
