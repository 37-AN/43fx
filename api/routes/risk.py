from __future__ import annotations

from fastapi import APIRouter, Depends

from api.models import RiskResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["risk"])


@router.get("/risk", response_model=RiskResponse)
def get_risk(container: ServiceContainer = Depends(get_container)) -> RiskResponse:
    metrics = container.runtime_service.refresh_account_metrics()
    config = container.config_service.load_copy()
    risk_cfg = config.get("risk", {})

    return RiskResponse(
        equity=float(metrics.get("equity", 0.0)),
        peak_equity=float(metrics.get("peak_equity", 0.0)),
        current_drawdown_pct=float(metrics.get("current_drawdown_pct", 0.0)),
        daily_realized_loss_pct=float(metrics.get("daily_loss_pct", 0.0)),
        risk_per_trade=float(risk_cfg.get("risk_per_trade", 0.01)),
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.03)),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.12)),
    )
