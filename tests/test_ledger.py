"""openai-token-monitor test suite — ported from KTH-Agent tests/test_spend_meter.py."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


# ---------------- pricing ----------------

def test_cost_estimate_table():
    from openai_token_monitor.pricing import estimate_cost_usd
    got = estimate_cost_usd("gpt-5-mini", 1_000_000, 1_000_000)
    assert abs(got - 2.25) < 1e-9
    got2 = estimate_cost_usd("gpt-4o-mini", 2_000_000, 500_000)
    assert abs(got2 - (0.30 + 0.30)) < 1e-9
    assert estimate_cost_usd("some-future-model", 100, 100) is None


# ---------------- config: cap default + ledger path ----------------

def test_daily_cap_default_is_zero_and_env_override(monkeypatch):
    from openai_token_monitor.config import default_daily_cap
    monkeypatch.delenv("OTM_DAILY_TOKEN_CAP", raising=False)
    assert default_daily_cap() == 0
    monkeypatch.setenv("OTM_DAILY_TOKEN_CAP", "42")
    assert default_daily_cap() == 42
    monkeypatch.setenv("OTM_DAILY_TOKEN_CAP", "not-an-int")
    assert default_daily_cap() == 0


def test_resolve_ledger_path_precedence(monkeypatch):
    from openai_token_monitor.config import DEFAULT_LEDGER_PATH, resolve_ledger_path
    monkeypatch.delenv("OTM_LEDGER_PATH", raising=False)
    assert resolve_ledger_path() == DEFAULT_LEDGER_PATH
    monkeypatch.setenv("OTM_LEDGER_PATH", "/tmp/otm-env.jsonl")
    assert str(resolve_ledger_path()) == "/tmp/otm-env.jsonl"
    assert str(resolve_ledger_path("/tmp/explicit.jsonl")) == "/tmp/explicit.jsonl"


# ---------------- SpendLedger: accumulation + flush format ----------------

def test_record_accumulates_calls_and_tokens():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 100, 20)
        ledger.record("gpt-4o-mini", 50, 10)
        ledger.record("gpt-4o", 5, 1)
        assert ledger.unflushed_calls() == 3
        assert ledger.unflushed_tokens() == (100 + 20 + 50 + 10 + 5 + 1)
        assert ledger._run_acc["gpt-4o-mini"].calls == 2
        assert ledger._run_acc["gpt-4o"].calls == 1


def test_flush_writes_expected_jsonl_shape():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 100, 20)
        ledger.flush()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert set(rec) == {"ts", "utc_date", "model", "calls", "prompt_tokens", "completion_tokens"}
        assert rec["utc_date"] == "2026-07-07"
        assert rec["model"] == "gpt-4o-mini"
        assert rec["calls"] == 1
        assert rec["prompt_tokens"] == 100
        assert rec["completion_tokens"] == 20


def test_flush_resets_unflushed_but_keeps_run_totals():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 100, 20)
        ledger.flush()
        assert ledger.unflushed_calls() == 0
        assert ledger.unflushed_tokens() == 0
        assert ledger._run_acc["gpt-4o-mini"].calls == 1
        assert ledger._run_acc["gpt-4o-mini"].prompt_tokens == 100


def test_flush_with_nothing_unflushed_is_a_noop():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.flush()
        assert not path.exists()


# ---------------- UTC-day bucketing + persistence + re-sum ----------------

def test_utc_day_bucketing_via_injected_date():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        day1 = SpendLedger(path=path, today_fn=lambda: "2026-07-06")
        day1.record("gpt-4o-mini", 1000, 200)
        day1.flush()
        day2 = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        day2.record("gpt-4o-mini", 5, 1)
        day2.flush()
        assert SpendLedger(path=path, today_fn=lambda: "2026-07-06").persisted_today_tokens() == 1200
        assert SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_tokens() == 6
        assert SpendLedger(path=path, today_fn=lambda: "2026-07-08").persisted_today_tokens() == 0


def test_resum_picks_up_concurrent_writers_lines():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        today = lambda: "2026-07-07"  # noqa: E731
        proc_a = SpendLedger(path=path, today_fn=today)
        proc_b = SpendLedger(path=path, today_fn=today)
        proc_a.record("gpt-4o-mini", 1000, 100)
        proc_a.flush()
        assert proc_b.today_total() == 1100
        proc_b.record("gpt-4o-mini", 10, 1)
        assert proc_b.today_total() == 1111
        assert proc_a.today_total() == 1100
        proc_b.flush()
        assert proc_a.persisted_today_tokens() == 1111
        assert proc_b.persisted_today_tokens() == 1111


def test_persisted_today_tokens_skips_a_torn_line():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        path.write_text(
            '{"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1, '
            '"prompt_tokens": 10, "completion_tokens": 5}\n'
            '{"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,'
            '\n'
            '\n',
            encoding="utf-8",
        )
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        assert ledger.persisted_today_tokens() == 15


def test_persisted_today_tokens_skips_valid_json_bad_type_fields():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        path.write_text(
            json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                        "prompt_tokens": "not-a-number", "completion_tokens": 5}) + "\n"
            + json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                          "prompt_tokens": 10, "completion_tokens": None}) + "\n"
            + json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                          "prompt_tokens": 7, "completion_tokens": 3}) + "\n",
            encoding="utf-8",
        )
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        assert ledger.persisted_today_tokens() == 20


def test_persisted_today_tokens_bad_type_line_excluded_atomically():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        path.write_text(
            json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                        "prompt_tokens": 1000, "completion_tokens": ["not", "a", "number"]}) + "\n",
            encoding="utf-8",
        )
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        assert ledger.persisted_today_tokens() == 0


def test_flush_failure_degrades_to_warning_and_keeps_counts(capsys):
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "not_a_dir"
        blocker.write_text("x", encoding="utf-8")
        bad_path = blocker / "spend.jsonl"
        ledger = SpendLedger(path=bad_path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 10, 5)
        ledger.flush()
        assert ledger.unflushed_calls() == 1
        assert ledger.unflushed_tokens() == 15


# ---------------- cap semantics ----------------

def test_cap_trips_exactly_at_boundary():
    from openai_token_monitor.ledger import DailyTokenCapReached, SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        cap = 1000
        ledger.record("gpt-4o-mini", 900, 99)
        ledger.check_cap(cap)
        assert ledger.tripped is False
        ledger.record("gpt-4o-mini", 1, 0)
        raised = False
        try:
            ledger.check_cap(cap)
        except DailyTokenCapReached as e:
            raised = True
            assert "1,000/1,000" in str(e)
            assert "OTM_DAILY_TOKEN_CAP" in str(e)
        assert raised is True
        assert ledger.tripped is True


def test_cap_trip_flushes_so_in_flight_result_is_not_discarded():
    from openai_token_monitor.ledger import DailyTokenCapReached, SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 1000, 0)
        try:
            ledger.check_cap(1000)
        except DailyTokenCapReached:
            pass
        reader = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        assert reader.persisted_today_tokens() == 1000


def test_cap_zero_disables_but_still_records():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 50_000_000, 0)
        ledger.check_cap(0)
        assert ledger.tripped is False
        assert ledger.unflushed_tokens() == 50_000_000


def test_check_cap_negative_also_disables():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 10, 0)
        ledger.check_cap(-1)


def test_daily_token_cap_reached_is_a_distinctive_exception_type():
    from openai_token_monitor.ledger import DailyTokenCapReached
    assert issubclass(DailyTokenCapReached, Exception)


# ---------------- per-model breakdown + summary_text + shared ledger ----------------

def test_persisted_today_by_model():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 900, 200)
        ledger.record("gpt-4o", 100, 34)
        ledger.flush()
        other = SpendLedger(path=path, today_fn=lambda: "2026-07-06")
        other.record("gpt-4o", 5000, 5000)
        other.flush()
        by_model = SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_by_model()
        assert by_model == {"gpt-4o-mini": (900, 200), "gpt-4o": (100, 34)}


def test_summary_text_contains_required_fields():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-5-mini", 1000, 200)
        ledger.record("gpt-4o-mini", 30, 10)
        text = ledger.summary_text(9_000_000)
        assert "gpt-5-mini" in text
        assert "calls=" in text and "in=" in text and "out=" in text
        assert "est=$" in text
        assert "UTC day total" in text
        assert "remaining today" in text
        assert "OpenAI dashboard remains the authority" in text


def test_summary_text_unknown_model_prints_tokens_only():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("some-unpriced-model", 100, 20)
        text = ledger.summary_text(9_000_000)
        assert "cost=n/a" in text
        assert "unpriced model" in text


def test_summary_text_cap_disabled():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-5-mini", 10, 5)
        text = ledger.summary_text(0)
        assert "cap disabled" in text
        assert "remaining today" not in text


def test_get_shared_ledger_is_singleton():
    from openai_token_monitor.ledger import get_shared_ledger
    assert get_shared_ledger() is get_shared_ledger()


# ---------------- fakes for the client wrapper (no network) ----------------

class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponsesUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeChatResponse:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 20, usage: bool = True):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens) if usage else None


class FakeOpenAI:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 20, usage: bool = True):
        self.calls = 0
        self._p = prompt_tokens
        self._c = completion_tokens
        self._usage = usage
        self.models = SimpleNamespace(marker="delegated")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.responses = SimpleNamespace(create=self._responses_create)

    def _chat_create(self, **kwargs):
        self.calls += 1
        return _FakeChatResponse(self._p, self._c, self._usage)

    def _responses_create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(usage=_FakeResponsesUsage(self._p, self._c))


def _make_metered(ledger, daily_cap=0, prompt_tokens=100, completion_tokens=20, usage=True):
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from openai_token_monitor.client import MeteredOpenAI
    m = MeteredOpenAI(ledger=ledger, daily_cap=daily_cap)
    fake = FakeOpenAI(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, usage=usage)
    m._bind(fake)
    return m, fake


# ---------------- MeteredOpenAI (sync) ----------------

def test_metered_records_usage_from_response():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        m, fake = _make_metered(ledger, daily_cap=0, prompt_tokens=321, completion_tokens=45)
        resp = m.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        assert resp.choices[0].message.content == "ok"
        assert fake.calls == 1
        assert ledger.unflushed_tokens() == 321 + 45
        assert ledger._run_acc["gpt-4o-mini"].calls == 1


def test_metered_blocked_by_cap_never_calls_api():
    from openai_token_monitor.ledger import DailyTokenCapReached, SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 10_000, 0)
        m, fake = _make_metered(ledger, daily_cap=10_000)
        raised = False
        try:
            m.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        except DailyTokenCapReached as e:
            raised = True
            assert "CAP REACHED" in str(e)
        assert raised is True
        assert fake.calls == 0
        assert ledger.tripped is True


def test_metered_flush_every_n_calls_triggers_mid_run():
    import openai_token_monitor.ledger as lmod
    from openai_token_monitor.ledger import SpendLedger
    orig = lmod.FLUSH_EVERY_N_CALLS
    lmod.FLUSH_EVERY_N_CALLS = 3
    try:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spend.jsonl"
            ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
            m, fake = _make_metered(ledger, daily_cap=0)
            for _ in range(5):
                m.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
            assert fake.calls == 5
            assert path.exists()
            assert ledger.unflushed_calls() == 2
            assert ledger._run_acc["gpt-4o-mini"].calls == 5
    finally:
        lmod.FLUSH_EVERY_N_CALLS = orig


def test_metered_accounting_error_does_not_crash_call(capsys):
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")

        def _boom(*a, **k):
            raise RuntimeError("simulated meter bug")

        ledger.record = _boom
        m, fake = _make_metered(ledger, daily_cap=0)
        resp = m.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert fake.calls == 1
        assert resp.choices[0].message.content == "ok"
        assert "spend-meter accounting error" in capsys.readouterr().out


def test_metered_delegates_unknown_attributes():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        m, fake = _make_metered(ledger, daily_cap=0)
        assert m.models.marker == "delegated"


def test_metered_records_responses_api_usage_fields():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        m, fake = _make_metered(ledger, daily_cap=0, prompt_tokens=70, completion_tokens=9)
        m.responses.create(model="gpt-4.1-mini", input="hello")
        assert ledger.unflushed_tokens() == 70 + 9
        assert ledger._run_acc["gpt-4.1-mini"].prompt_tokens == 70


def test_metered_no_usage_is_skipped_not_an_error():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        m, fake = _make_metered(ledger, daily_cap=0, usage=False)
        m.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert fake.calls == 1
        assert ledger.unflushed_tokens() == 0


# ---------------- MeteredAsyncOpenAI ----------------

class FakeAsyncOpenAI:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 20):
        self.calls = 0
        self._p = prompt_tokens
        self._c = completion_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.responses = SimpleNamespace(create=self._responses_create)

    async def _chat_create(self, **kwargs):
        self.calls += 1
        return _FakeChatResponse(self._p, self._c)

    async def _responses_create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(usage=_FakeResponsesUsage(self._p, self._c))


def _make_async_metered(ledger, daily_cap=0, prompt_tokens=100, completion_tokens=20):
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from openai_token_monitor.client import MeteredAsyncOpenAI
    m = MeteredAsyncOpenAI(ledger=ledger, daily_cap=daily_cap)
    fake = FakeAsyncOpenAI(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    m._bind(fake)
    return m, fake


def test_async_metered_records_usage():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        m, fake = _make_async_metered(ledger, daily_cap=0, prompt_tokens=200, completion_tokens=30)

        async def _go():
            return await m.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        resp = asyncio.run(_go())
        assert resp.choices[0].message.content == "ok"
        assert fake.calls == 1
        assert ledger.unflushed_tokens() == 230


def test_async_metered_blocked_by_cap_never_calls_api():
    from openai_token_monitor.ledger import DailyTokenCapReached, SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 10_000, 0)
        m, fake = _make_async_metered(ledger, daily_cap=10_000)

        async def _go():
            return await m.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        raised = False
        try:
            asyncio.run(_go())
        except DailyTokenCapReached:
            raised = True
        assert raised is True
        assert fake.calls == 0
        assert ledger.tripped is True


# ---------------- CLI: status formatting + commands ----------------

def test_humanize():
    from openai_token_monitor.cli import _humanize
    assert _humanize(0) == "0"
    assert _humanize(999) == "999"
    assert _humanize(900_000) == "900k"
    assert _humanize(1_100_000) == "1.1M"


def test_format_status_with_cap():
    from openai_token_monitor.cli import format_status
    from openai_token_monitor.pricing import estimate_cost_usd
    cost = estimate_cost_usd("gpt-4o-mini", 900_000, 200_000)
    text = format_status({"gpt-4o-mini": (900_000, 200_000)}, total=1_100_000,
                         cap=9_000_000, cost=cost, any_unpriced=False)
    lines = text.splitlines()
    assert lines[0] == (
        f"spend ledger: 1,100,000 / 9,000,000 tokens today (UTC) — "
        f"7,900,000 remaining (~${cost:.2f} est.)"
    )
    assert "gpt-4o-mini" in lines[1]
    assert "1,100,000 tokens" in lines[1]
    assert "(900k prompt / 200k completion)" in lines[1]


def test_format_status_cap_disabled():
    from openai_token_monitor.cli import format_status
    text = format_status({"gpt-4o": (100, 34)}, total=134, cap=0)
    assert text.splitlines()[0].startswith("spend ledger: 134 tokens today (UTC) — cap disabled")


def test_cmd_path_prints_resolved_path(capsys, monkeypatch):
    from openai_token_monitor.cli import cmd_path
    monkeypatch.setenv("OTM_LEDGER_PATH", "/tmp/otm-cli.jsonl")
    rc = cmd_path(SimpleNamespace())
    assert rc == 0
    assert capsys.readouterr().out.strip() == "/tmp/otm-cli.jsonl"


def test_cmd_status_smoke(capsys, monkeypatch):
    from openai_token_monitor.cli import cmd_status
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        from openai_token_monitor.ledger import SpendLedger
        import openai_token_monitor.ledger as lmod
        seed = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        seed.record("gpt-4o-mini", 900_000, 200_000)
        seed.flush()
        monkeypatch.setattr(lmod, "_SHARED_LEDGER",
                            SpendLedger(path=path, today_fn=lambda: "2026-07-07"))
        monkeypatch.setenv("OTM_DAILY_TOKEN_CAP", "9000000")
        rc = cmd_status(SimpleNamespace(watch=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert "spend ledger:" in out
        assert "gpt-4o-mini" in out


# ---------------- public API surface ----------------

def test_public_exports():
    import openai_token_monitor as otm
    for name in ("MeteredOpenAI", "MeteredAsyncOpenAI", "DailyTokenCapReached",
                 "SpendLedger", "get_shared_ledger", "estimate_cost_usd", "__version__"):
        assert hasattr(otm, name), f"missing export: {name}"
    from openai_token_monitor import get_shared_ledger, DailyTokenCapReached
    assert issubclass(DailyTokenCapReached, Exception)
    assert get_shared_ledger() is get_shared_ledger()


# ================= user-supplied pricing (new feature) =================

def test_estimate_cost_usd_supplied_prices_win():
    from openai_token_monitor.pricing import estimate_cost_usd
    got = estimate_cost_usd("my-custom-model", 1_000_000, 1_000_000,
                            input_price=1.00, output_price=3.00)
    assert abs(got - 4.00) < 1e-9
    got2 = estimate_cost_usd("gpt-4o-mini", 1_000_000, 0, input_price=9.99)
    assert abs(got2 - 9.99) < 1e-9


def test_estimate_cost_usd_one_sided_unknown_is_none():
    from openai_token_monitor.pricing import estimate_cost_usd
    assert estimate_cost_usd("my-custom-model", 100, 100, input_price=1.00) is None
    assert estimate_cost_usd("my-custom-model", 100, 100) is None


def test_record_stores_price_and_flush_writes_it():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 100, 20, input_price=0.15, output_price=0.60)
        ledger.flush()
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["input_price"] == 0.15
        assert rec["output_price"] == 0.60


def test_record_without_price_omits_keys():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 100, 20)
        ledger.flush()
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "input_price" not in rec and "output_price" not in rec


def test_accumulator_remembers_last_supplied_price():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("gpt-4o-mini", 10, 2, input_price=0.15, output_price=0.60)
        ledger.record("gpt-4o-mini", 10, 2)
        assert ledger._acc["gpt-4o-mini"].input_price == 0.15
        assert ledger._acc["gpt-4o-mini"].output_price == 0.60


def test_persisted_today_cost_uses_stored_prices_for_unknown_model():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        ledger.record("my-custom-model", 1_000_000, 1_000_000,
                      input_price=1.00, output_price=3.00)
        ledger.flush()
        total, any_unpriced = SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_cost()
        assert abs(total - 4.00) < 1e-9
        assert any_unpriced is False


def test_persisted_today_cost_sums_differing_prices_per_line():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        path.write_text(
            json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                        "prompt_tokens": 1_000_000, "completion_tokens": 0,
                        "input_price": 1.00, "output_price": 2.00}) + "\n"
            + json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "m", "calls": 1,
                          "prompt_tokens": 1_000_000, "completion_tokens": 0,
                          "input_price": 5.00, "output_price": 2.00}) + "\n",
            encoding="utf-8",
        )
        total, any_unpriced = SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_cost()
        assert abs(total - 6.00) < 1e-9
        assert any_unpriced is False


def test_persisted_today_cost_flags_unpriced_lines():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        path.write_text(
            json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "gpt-4o-mini", "calls": 1,
                        "prompt_tokens": 1_000_000, "completion_tokens": 0}) + "\n"
            + json.dumps({"ts": "x", "utc_date": "2026-07-07", "model": "mystery", "calls": 1,
                          "prompt_tokens": 500, "completion_tokens": 500}) + "\n",
            encoding="utf-8",
        )
        total, any_unpriced = SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_cost()
        assert abs(total - 0.15) < 1e-9
        assert any_unpriced is True


def test_summary_text_uses_client_supplied_price():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        ledger = SpendLedger(path=Path(d) / "spend.jsonl", today_fn=lambda: "2026-07-07")
        ledger.record("brand-new-model", 1_000_000, 0, input_price=2.00, output_price=8.00)
        text = ledger.summary_text(0)
        assert "brand-new-model" in text
        assert "est=$2.0000" in text


def _make_metered_priced(ledger, input_price, output_price, daily_cap=0,
                         prompt_tokens=100, completion_tokens=20):
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from openai_token_monitor.client import MeteredOpenAI
    m = MeteredOpenAI(ledger=ledger, daily_cap=daily_cap,
                      input_price=input_price, output_price=output_price)
    fake = FakeOpenAI(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    m._bind(fake)
    return m, fake


def test_metered_writes_supplied_prices_to_ledger():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        m, fake = _make_metered_priced(ledger, input_price=2.00, output_price=8.00,
                                       prompt_tokens=1_000_000, completion_tokens=0)
        m.chat.completions.create(model="brand-new-model",
                                  messages=[{"role": "user", "content": "hi"}])
        assert m.input_price == 2.00 and m.output_price == 8.00
        ledger.flush()
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["input_price"] == 2.00 and rec["output_price"] == 8.00
        total, any_unpriced = SpendLedger(path=path, today_fn=lambda: "2026-07-07").persisted_today_cost()
        assert abs(total - 2.00) < 1e-9 and any_unpriced is False


def test_async_metered_writes_supplied_prices_to_ledger():
    from openai_token_monitor.ledger import SpendLedger
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        ledger = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from openai_token_monitor.client import MeteredAsyncOpenAI
        m = MeteredAsyncOpenAI(ledger=ledger, daily_cap=0, input_price=1.5, output_price=6.0)
        fake = FakeAsyncOpenAI(prompt_tokens=1_000_000, completion_tokens=0)
        m._bind(fake)

        async def _go():
            return await m.chat.completions.create(
                model="brand-new-model", messages=[{"role": "user", "content": "hi"}])

        asyncio.run(_go())
        assert m.input_price == 1.5 and m.output_price == 6.0
        ledger.flush()
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["input_price"] == 1.5 and rec["output_price"] == 6.0


def test_format_status_unpriced_note():
    from openai_token_monitor.cli import format_status
    text = format_status({"m": (10, 5)}, total=15, cap=9_000_000,
                         cost=1.23, any_unpriced=True)
    assert "(~$1.23 est.; some models unpriced)" in text.splitlines()[0]


def test_cmd_status_shows_dollars_for_custom_priced_model(capsys, monkeypatch):
    from openai_token_monitor.cli import cmd_status
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "spend.jsonl"
        from openai_token_monitor.ledger import SpendLedger
        import openai_token_monitor.ledger as lmod
        seed = SpendLedger(path=path, today_fn=lambda: "2026-07-07")
        seed.record("my-custom-model", 1_000_000, 0, input_price=2.00, output_price=8.00)
        seed.flush()
        monkeypatch.setattr(lmod, "_SHARED_LEDGER",
                            SpendLedger(path=path, today_fn=lambda: "2026-07-07"))
        monkeypatch.setenv("OTM_DAILY_TOKEN_CAP", "9000000")
        rc = cmd_status(SimpleNamespace(watch=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert "~$2.00 est." in out
        assert "my-custom-model" in out


# ---- dual-run footer ----
class _MonkeyPatch:
    def __init__(self):
        self._env = []
        self._attrs = []

    def setenv(self, k, v):
        self._env.append((k, os.environ.get(k)))
        os.environ[k] = v

    def delenv(self, k, raising=False):
        self._env.append((k, os.environ.get(k)))
        os.environ.pop(k, None)

    def setattr(self, target, name, value):
        self._attrs.append((target, name, getattr(target, name, None), hasattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, old, existed in reversed(self._attrs):
            if existed:
                setattr(target, name, old)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        for k, old in reversed(self._env):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _run_all():
    import contextlib
    import inspect
    import io

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        params = inspect.signature(fn).parameters
        kwargs = {}
        mp = _MonkeyPatch() if "monkeypatch" in params else None
        if mp is not None:
            kwargs["monkeypatch"] = mp
        buf = io.StringIO() if "capsys" in params else None
        try:
            if buf is not None:
                with contextlib.redirect_stdout(buf):
                    class _Capsys:
                        def readouterr(self_inner):
                            return SimpleNamespace(out=buf.getvalue())
                    kwargs["capsys"] = _Capsys()
                    fn(**kwargs)
            else:
                fn(**kwargs)
        finally:
            if mp is not None:
                mp.undo()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
