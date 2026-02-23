from __future__ import annotations

from fastapi import APIRouter, Depends

from api.models import ConfigUpdateRequest, ConfigUpdateResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["config"])


@router.post("/config/update", response_model=ConfigUpdateResponse)
def update_config(
    payload: ConfigUpdateRequest,
    container: ServiceContainer = Depends(get_container),
) -> ConfigUpdateResponse:
    applied = container.config_service.safe_update(payload.model_dump())
    container.trading_service.apply_runtime_risk_updates(applied)
    return ConfigUpdateResponse(status="ok", applied=applied)
