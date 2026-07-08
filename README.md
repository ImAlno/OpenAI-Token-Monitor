# openai-token-monitor

A tiny, dependency-light **guardrail** that counts your OpenAI token usage and can stop
you before you blow past a daily (UTC) token cap.

> **Honest framing — read this first.** This is a *self-metering guardrail*, **not** a
> billing dashboard. It counts **only** the OpenAI calls you route through it (reading
> `usage` off each response), buckets them by UTC day in a local append-only JSONL
> ledger, and — if you set a cap — refuses the next call that would exceed it. It does
> **not** know what other keys, apps, or teammates spend. The OpenAI dashboard remains
> the authority for org-wide truth.

## Install

```bash
pip install openai-token-monitor
```

## Quickstart

```python
from openai_token_monitor import MeteredOpenAI

# input_price / output_price are USD per 1M tokens (copy from platform.openai.com/pricing)
client = MeteredOpenAI(daily_cap=9_000_000, input_price=0.15, output_price=0.60)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)   # cap checked BEFORE the call; usage + your prices recorded AFTER
```

Then check today's usage from the terminal:

```
$ otm status
spend ledger: 1,234,567 / 9,000,000 tokens today (UTC) — 7,765,433 remaining (~$0.31 est.)
  gpt-4o-mini      1,100,000 tokens   (900k prompt / 200k completion)
  gpt-4o             134,567 tokens   (100k prompt / 34k completion)
```

Set `OPENAI_API_KEY` as usual — the `openai` SDK reads it; this package does not touch it.

## The "free daily allowance" and your cap

OpenAI grants some accounts a **free daily token allowance** (e.g. via the data-sharing
program). It is **per-account and changes over time**, so this tool ships with the cap
**disabled** (`0`) — you get monitoring immediately and opt into enforcement when you know
your own number:

- Find your allowance/limits at <https://platform.openai.com/settings/organization/limits>.
- Set a cap a hair under it so this tool alone can never tip you into paid usage.

```bash
export OTM_DAILY_TOKEN_CAP=9000000     # enforce; 0 (default) = monitor only
```

or per-client: `MeteredOpenAI(daily_cap=9_000_000)`.

When the cap is reached, the next metered call raises `DailyTokenCapReached` **before** it
fires — the in-flight call that tipped you over is still recorded; the *next* one is blocked.

```python
from openai_token_monitor import DailyTokenCapReached
try:
    client.chat.completions.create(model="gpt-4o-mini", messages=[...])
except DailyTokenCapReached as e:
    print(e)   # DAILY OPENAI TOKEN CAP REACHED (…/…) — resume after 00:00 UTC or raise OTM_DAILY_TOKEN_CAP
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OTM_DAILY_TOKEN_CAP` | `0` (disabled) | Daily UTC token cap; `0` or negative = monitor only. |
| `OTM_LEDGER_PATH` | `~/.openai-token-monitor/spend.jsonl` | Where the ledger is written. |
| `OPENAI_API_KEY` | — | Read by the `openai` SDK (not by this package). |

Kwargs on `MeteredOpenAI` / `MeteredAsyncOpenAI` override env: `daily_cap=`, `ledger_path=`,
`ledger=`, `input_price=`, `output_price=`. All other kwargs (`api_key=`, `base_url=`,
`organization=`, …) pass through to `openai.OpenAI`.

## Custom pricing

Pass your own prices so the `$` estimate is exact — no waiting on a package update, and it
works for fine-tuned or brand-new models the built-in table has never heard of:

```python
client = MeteredOpenAI(
    daily_cap=9_000_000,
    input_price=0.15,    # USD per 1M input (prompt) tokens
    output_price=0.60,   # USD per 1M output (completion) tokens
)
```

The prices you pass are written into each ledger line, so `otm status` (a separate process)
reports real dollars for that model with no table lookup. One price pair applies to every
model called through the client — use a second client for a differently-priced model, or
omit the prices to let the built-in table handle known models. `SpendLedger.record(model,
prompt_tokens, completion_tokens, input_price=..., output_price=...)` accepts the same pair
for the low-level API.

## Low-level ledger (explicit style)

```python
from openai_token_monitor import SpendLedger, get_shared_ledger, estimate_cost_usd

ledger = get_shared_ledger()
ledger.record("gpt-4o-mini", prompt_tokens=123, completion_tokens=45)
ledger.check_cap(9_000_000)          # raises DailyTokenCapReached if today's total >= cap
print(ledger.today_total())          # persisted (cross-process) + this process's unflushed
```

`MeteredAsyncOpenAI` is the drop-in for `openai.AsyncOpenAI()` with `await`able calls.

## CLI

- `otm status` — today's usage vs. cap, remaining, estimated $, per-model breakdown.
  `otm status --watch [SECONDS]` re-prints on an interval (default 2s) until Ctrl-C.
- `otm path` — print the resolved ledger file path.

(`openai-token-monitor` is an alias for `otm`.)

## Limitations

- Counts **only** calls routed through this client (or `SpendLedger.record`). Not org-wide.
- **Streaming:** usage is recorded only if the response carries it (needs
  `stream_options={"include_usage": True}`); otherwise it is silently skipped. v1 targets
  non-streaming; streaming is best-effort.
- **Set your prices for exact `$`.** The built-in table
  (`openai_token_monitor.pricing.OPENAI_PRICE_PER_MILLION`) intentionally covers only a
  couple of verified models — pass `input_price`/`output_price` to `MeteredOpenAI` (or
  `SpendLedger.record`) for anything else. Models with no supplied price and no table
  entry are tracked in tokens with no `$` (and `otm status` says so).

## License

MIT.
