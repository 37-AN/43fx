from __future__ import annotations

from dataclasses import dataclass

from .ai_service import AIService
from .backtest_service import BacktestService
from .config_service import ConfigService
from .runtime_service import RuntimeService
from .state_manager import RuntimeStateManager
from .trading_control_service import TradingControlService


@dataclass
class ServiceContainer:
    config_service: ConfigService
    state: RuntimeStateManager
    runtime_service: RuntimeService
    trading_service: TradingControlService
    ai_service: AIService
    backtest_service: BacktestService


def build_container(config_path: str = "config/settings.yaml") -> ServiceContainer:
    config_service = ConfigService(config_path=config_path)
    state = RuntimeStateManager()
    runtime_service = RuntimeService(config_service=config_service, state=state)
    trading_service = TradingControlService(config_service=config_service, state=state)
    ai_service = AIService(config_service=config_service, state=state)
    backtest_service = BacktestService(config_service=config_service, state=state)
    return ServiceContainer(
        config_service=config_service,
        state=state,
        runtime_service=runtime_service,
        trading_service=trading_service,
        ai_service=ai_service,
        backtest_service=backtest_service,
    )
