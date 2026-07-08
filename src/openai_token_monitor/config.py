"""Configuration: the default ledger location and the env-var readers. No monorepo path
assumptions — the ledger is machine-global under the user's home directory."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LEDGER_DIR = Path.home() / ".openai-token-monitor"
DEFAULT_LEDGER_PATH = DEFAULT_LEDGER_DIR / "spend.jsonl"


def default_daily_cap() -> int:
    """The daily UTC token cap when none is passed explicitly. Reads OTM_DAILY_TOKEN_CAP;
    default 0 = disabled (pure monitoring). A non-integer value degrades to 0 (never
    raises — a bad env var must not crash the user's client construction)."""
    raw = os.environ.get("OTM_DAILY_TOKEN_CAP", "0")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def resolve_ledger_path(ledger_path=None) -> Path:
    """Resolve the ledger file path: explicit arg > OTM_LEDGER_PATH env > default
    (~/.openai-token-monitor/spend.jsonl)."""
    if ledger_path is not None:
        return Path(ledger_path)
    env = os.environ.get("OTM_LEDGER_PATH")
    if env:
        return Path(env)
    return DEFAULT_LEDGER_PATH
