from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.models import TradeItem, TradesResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["trades"])


@router.get("/trades", response_model=TradesResponse)
def get_trades(
    limit: int = Query(default=100, ge=1, le=1000),
    container: ServiceContainer = Depends(get_container),
) -> TradesResponse:
    rows = container.runtime_service.get_trade_history(limit=limit)
    return TradesResponse(trades=[TradeItem(**row) for row in rows])
