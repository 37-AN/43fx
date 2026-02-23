from __future__ import annotations

from fastapi import APIRouter, Depends

from api.models import EquityPoint, EquityResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["equity"])


@router.get("/equity", response_model=EquityResponse)
def get_equity(container: ServiceContainer = Depends(get_container)) -> EquityResponse:
    points = container.runtime_service.get_equity_points()
    return EquityResponse(points=[EquityPoint(**row) for row in points])
