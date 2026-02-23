from __future__ import annotations

from fastapi import APIRouter, Depends

from api.models import StatusResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status(container: ServiceContainer = Depends(get_container)) -> StatusResponse:
    metrics = container.runtime_service.refresh_account_metrics()
    config = container.config_service.load_copy()
    ai_status = container.ai_service.status()

    return StatusResponse(
        live_mode_running=container.state.is_live_running(),
        broker_type=str(metrics.get("broker_type", "unknown")),
        current_equity=float(metrics.get("equity", 0.0)),
        current_drawdown_pct=float(metrics.get("current_drawdown_pct", 0.0)),
        daily_loss_pct=float(metrics.get("daily_loss_pct", 0.0)),
        ai_enabled=bool(ai_status.get("enabled", False)),
        open_positions=int(metrics.get("open_positions", 0)),
        last_trade_time=container.runtime_service.get_last_trade_time(),
        system_uptime=container.state.snapshot().get("system_uptime", "0:00:00"),
    )
