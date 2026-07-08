"""Persistent, cross-process, per-UTC-day OpenAI token ledger (stdlib only)."""
from __future__ import annotations

import atexit
import json
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai_token_monitor.config import resolve_ledger_path
from openai_token_monitor.pricing import estimate_cost_usd

FLUSH_EVERY_N_CALLS = 200


class DailyTokenCapReached(RuntimeError):
    """Raised by SpendLedger.check_cap when today's UTC token total already meets/exceeds
    the cap."""


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class _SpendAccumulator:
    """Calls/tokens for one model bucket, plus the price pair it was last metered under
    (USD per 1M tokens; None = not supplied, price via the table instead)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_price: float | None = None
    output_price: float | None = None

    def add(self, prompt_tokens, completion_tokens, input_price=None, output_price=None):
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if input_price is not None:
            self.input_price = input_price
        if output_price is not None:
            self.output_price = output_price

    def reset(self):
        # Zero the counters only; the remembered price is stable across flushes.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


class SpendLedger:
    def __init__(self, path: Path | None = None, today_fn=_utc_today):
        self.path = Path(path) if path is not None else resolve_ledger_path()
        self._today_fn = today_fn
        self._acc: dict[str, _SpendAccumulator] = {}
        self._run_acc: dict[str, _SpendAccumulator] = {}
        self.tripped = False
        self.trip_message = None

    @staticmethod
    def _bucket(store: dict, model: str) -> _SpendAccumulator:
        return store.setdefault(model, _SpendAccumulator())

    def record(self, model, prompt_tokens, completion_tokens,
               input_price=None, output_price=None):
        self._bucket(self._acc, model).add(prompt_tokens, completion_tokens,
                                           input_price, output_price)
        self._bucket(self._run_acc, model).add(prompt_tokens, completion_tokens,
                                               input_price, output_price)

    def unflushed_tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self._acc.values())

    def unflushed_calls(self) -> int:
        return sum(a.calls for a in self._acc.values())

    def persisted_today_tokens(self) -> int:
        today = self._today_fn()
        total = 0
        if not self.path.exists():
            return 0
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if rec.get("utc_date") != today:
                        continue
                    try:
                        line_tokens = (
                            int(rec.get("prompt_tokens", 0) or 0)
                            + int(rec.get("completion_tokens", 0) or 0)
                        )
                    except (ValueError, TypeError):
                        continue
                    total += line_tokens
        except OSError as e:
            print(f"  ⚠ spend-meter: could not fully read ledger ({e}); "
                  f"using partial today-total {total:,}")
        return total

    def today_total(self) -> int:
        return self.persisted_today_tokens() + self.unflushed_tokens()

    def persisted_today_by_model(self) -> dict[str, tuple[int, int]]:
        today = self._today_fn()
        out: dict[str, list[int]] = {}
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if rec.get("utc_date") != today:
                        continue
                    try:
                        p = int(rec.get("prompt_tokens", 0) or 0)
                        c = int(rec.get("completion_tokens", 0) or 0)
                    except (ValueError, TypeError):
                        continue
                    bucket = out.setdefault(str(rec.get("model", "unknown")), [0, 0])
                    bucket[0] += p
                    bucket[1] += c
        except OSError:
            pass
        return {m: (p, c) for m, (p, c) in out.items()}

    def persisted_today_cost(self) -> tuple[float, bool]:
        """Total estimated USD for today's persisted lines, summed PER LINE so per-line
        price differences add up correctly. Returns (total_usd, any_unpriced). A line is
        'unpriced' when neither a stored price nor the table can resolve both sides.
        Robust to corrupt token/price fields (a bad line is skipped)."""
        today = self._today_fn()
        total = 0.0
        any_unpriced = False
        if not self.path.exists():
            return (0.0, False)

        def _px(v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if rec.get("utc_date") != today:
                        continue
                    try:
                        p = int(rec.get("prompt_tokens", 0) or 0)
                        c = int(rec.get("completion_tokens", 0) or 0)
                    except (ValueError, TypeError):
                        continue
                    cost = estimate_cost_usd(
                        str(rec.get("model", "")), p, c,
                        _px(rec.get("input_price")), _px(rec.get("output_price")),
                    )
                    if cost is None:
                        any_unpriced = True
                    else:
                        total += cost
        except OSError:
            pass
        return (total, any_unpriced)

    def flush(self) -> None:
        lines = [(model, acc) for model, acc in self._acc.items() if acc.calls]
        if not lines:
            return
        today = self._today_fn()
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for model, acc in lines:
                    rec = {
                        "ts": ts, "utc_date": today, "model": model, "calls": acc.calls,
                        "prompt_tokens": acc.prompt_tokens,
                        "completion_tokens": acc.completion_tokens,
                    }
                    if acc.input_price is not None:
                        rec["input_price"] = acc.input_price
                    if acc.output_price is not None:
                        rec["output_price"] = acc.output_price
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"  ⚠ spend-meter: failed to flush ledger ({e}); "
                  f"unflushed totals kept for the next flush attempt")
            return
        for _model, acc in lines:
            acc.reset()

    def check_cap(self, cap: int) -> None:
        if cap <= 0:
            return
        try:
            total = self.today_total()
        except Exception as e:
            print(f"  ⚠ spend-meter: could not compute today's total ({e}); cap check skipped")
            return
        if total >= cap:
            self.trip_message = (
                f"DAILY OPENAI TOKEN CAP REACHED ({total:,}/{cap:,}) — resume after "
                f"00:00 UTC or raise OTM_DAILY_TOKEN_CAP"
            )
            self.tripped = True
            self.flush()
            raise DailyTokenCapReached(self.trip_message)

    def summary_text(self, cap: int) -> str:
        self.flush()
        lines = ["─" * 64, "  OPENAI TOKEN SPEND", "─" * 64]
        run_calls = run_in = run_out = 0
        run_cost = 0.0
        unpriced = False
        for model, acc in sorted(self._run_acc.items()):
            run_calls += acc.calls
            run_in += acc.prompt_tokens
            run_out += acc.completion_tokens
            cost = estimate_cost_usd(model, acc.prompt_tokens, acc.completion_tokens,
                                     acc.input_price, acc.output_price)
            if cost is None:
                unpriced = True
                lines.append(
                    f"  {model:<16} calls={acc.calls:<5} in={acc.prompt_tokens:>9,} "
                    f"out={acc.completion_tokens:>8,}  cost=n/a (unpriced model)"
                )
            else:
                run_cost += cost
                lines.append(
                    f"  {model:<16} calls={acc.calls:<5} in={acc.prompt_tokens:>9,} "
                    f"out={acc.completion_tokens:>8,}  est=${cost:.4f}"
                )
        if not self._run_acc:
            lines.append("  (no calls recorded this run)")
        else:
            note = " (+ unpriced model tokens above not included)" if unpriced else ""
            lines.append(
                f"  {'this run':<16} calls={run_calls:<5} in={run_in:>9,} "
                f"out={run_out:>8,}  est=${run_cost:.4f}{note}"
            )
        day_total = self.persisted_today_tokens()
        if cap > 0:
            remaining = max(0, cap - day_total)
            lines.append(f"  UTC day total  : {day_total:,} / {cap:,} tokens ({day_total / cap * 100:.1f}%)")
            lines.append(f"  remaining today: {remaining:,} tokens")
        else:
            lines.append(f"  UTC day total  : {day_total:,} tokens (cap disabled)")
        if self.tripped and self.trip_message:
            lines.append(f"  ✖ {self.trip_message}")
        lines.append("  NOTE: counts only calls routed through openai-token-monitor — the")
        lines.append("        OpenAI dashboard remains the authority (other keys/apps spend too).")
        lines.append("─" * 64)
        return "\n".join(lines)


_SHARED_LEDGER: SpendLedger | None = None
_ATEXIT_LEDGERS: "weakref.WeakSet[SpendLedger]" = weakref.WeakSet()
_ATEXIT_HOOKED = False


def _flush_all_atexit() -> None:
    for led in list(_ATEXIT_LEDGERS):
        try:
            led.flush()
        except Exception:
            pass


def register_atexit_flush(ledger: SpendLedger) -> None:
    global _ATEXIT_HOOKED
    _ATEXIT_LEDGERS.add(ledger)
    if not _ATEXIT_HOOKED:
        atexit.register(_flush_all_atexit)
        _ATEXIT_HOOKED = True


def get_shared_ledger() -> SpendLedger:
    global _SHARED_LEDGER
    if _SHARED_LEDGER is None:
        _SHARED_LEDGER = SpendLedger()
        register_atexit_flush(_SHARED_LEDGER)
    return _SHARED_LEDGER
