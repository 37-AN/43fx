from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    live_mode_running: bool
    broker_type: str
    current_equity: float
    current_drawdown_pct: float
    daily_loss_pct: float
    ai_enabled: bool
    open_positions: int
    last_trade_time: Optional[str] = None
    system_uptime: str


class RiskResponse(BaseModel):
    equity: float
    peak_equity: float
    current_drawdown_pct: float
    daily_realized_loss_pct: float
    risk_per_trade: float
    max_daily_loss_pct: float
    max_drawdown_pct: float


class TradeItem(BaseModel):
    direction: str
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    ai_probability: Optional[float] = None
    size: Optional[float] = None
    entry: Optional[float] = None
    exit: Optional[float] = None


class TradesResponse(BaseModel):
    trades: List[TradeItem]


class EquityPoint(BaseModel):
    timestamp: str
    equity: float
    drawdown_pct: float


class EquityResponse(BaseModel):
    points: List[EquityPoint]


class ActionResponse(BaseModel):
    status: str
    detail: str


class AITrainResponse(BaseModel):
    status: str
    trained: bool
    samples: int
    positives: int
    negatives: int
    training_accuracy: float
    model_path: str


class AIStatusResponse(BaseModel):
    enabled: bool
    model_loaded: bool
    model_path: str
    probability_threshold: float
    training_sample_size: int
    class_balance: Dict[str, int]


class BacktestRunRequest(BaseModel):
    instrument: str = Field(default="EURUSD")
    start_date: date
    end_date: date
    use_synthetic: bool = False


class BacktestRunResponse(BaseModel):
    status: str
    summary_metrics: Dict[str, float]


class ConfigUpdateRequest(BaseModel):
    risk_per_trade: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_daily_loss_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_drawdown_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ai_probability_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConfigUpdateResponse(BaseModel):
    status: str
    applied: Dict[str, Any]
