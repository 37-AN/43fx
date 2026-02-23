from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.models import BacktestRunRequest, BacktestRunResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["backtest"])


@router.post("/backtest/run", response_model=BacktestRunResponse)
async def run_backtest_endpoint(
    payload: BacktestRunRequest,
    container: ServiceContainer = Depends(get_container),
) -> BacktestRunResponse:
    if payload.end_date < payload.start_date:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")

    try:
        summary = await asyncio.to_thread(
            container.backtest_service.run,
            payload.instrument,
            payload.start_date,
            payload.end_date,
            payload.use_synthetic,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return BacktestRunResponse(status="ok", summary_metrics=summary)
