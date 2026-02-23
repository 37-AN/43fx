from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from api.models import AIStatusResponse, AITrainResponse, ActionResponse
from api.services.container import ServiceContainer

from .deps import get_container

router = APIRouter(tags=["ai"])


@router.get("/ai/status", response_model=AIStatusResponse)
def ai_status(container: ServiceContainer = Depends(get_container)) -> AIStatusResponse:
    return AIStatusResponse(**container.ai_service.status())


@router.post("/ai/train", response_model=AITrainResponse)
async def ai_train(container: ServiceContainer = Depends(get_container)) -> AITrainResponse:
    summary = await asyncio.to_thread(container.ai_service.train)
    return AITrainResponse(**summary)


@router.post("/ai/enable", response_model=ActionResponse)
def ai_enable(container: ServiceContainer = Depends(get_container)) -> ActionResponse:
    summary = container.ai_service.set_enabled(True)
    if not summary["model_loaded"]:
        return ActionResponse(status="warning", detail="AI model missing. Train model before enabling.")
    return ActionResponse(status="ok", detail="AI filter enabled")


@router.post("/ai/disable", response_model=ActionResponse)
def ai_disable(container: ServiceContainer = Depends(get_container)) -> ActionResponse:
    container.ai_service.set_enabled(False)
    return ActionResponse(status="ok", detail="AI filter disabled")
