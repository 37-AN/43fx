from __future__ import annotations

from fastapi import APIRouter, Depends

from api.models import ActionResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["strategy"])


@router.post("/strategy/start", response_model=ActionResponse)
def start_strategy(container: ServiceContainer = Depends(get_container)) -> ActionResponse:
    result = container.trading_service.start()
    return ActionResponse(**result)


@router.post("/strategy/stop", response_model=ActionResponse)
def stop_strategy(container: ServiceContainer = Depends(get_container)) -> ActionResponse:
    result = container.trading_service.stop()
    return ActionResponse(**result)


@router.post("/strategy/restart", response_model=ActionResponse)
def restart_strategy(container: ServiceContainer = Depends(get_container)) -> ActionResponse:
    result = container.trading_service.restart()
    return ActionResponse(**result)
