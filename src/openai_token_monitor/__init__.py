"""openai-token-monitor: a self-metering guardrail for OpenAI token usage."""
from __future__ import annotations

from openai_token_monitor.client import MeteredAsyncOpenAI, MeteredOpenAI
from openai_token_monitor.ledger import (
    DailyTokenCapReached,
    SpendLedger,
    get_shared_ledger,
)
from openai_token_monitor.pricing import estimate_cost_usd

__version__ = "0.1.0"

__all__ = [
    "MeteredOpenAI",
    "MeteredAsyncOpenAI",
    "DailyTokenCapReached",
    "SpendLedger",
    "get_shared_ledger",
    "estimate_cost_usd",
    "__version__",
]
