"""Command-line monitor. `otm status` / `otm path`. stdlib argparse only."""
from __future__ import annotations

import argparse
import time

from openai_token_monitor.config import default_daily_cap, resolve_ledger_path
from openai_token_monitor.ledger import get_shared_ledger


def _humanize(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


def format_status(by_model: dict, total: int, cap: int, cost: float = 0.0,
                  any_unpriced: bool = False) -> str:
    """Render the `otm status` snapshot. `cost`/`any_unpriced` come precomputed from the
    ledger's stored per-line prices (SpendLedger.persisted_today_cost); the built-in table
    is only a fallback there. Per-model breakdown lines are tokens-only."""
    if cost > 0:
        note = "; some models unpriced" if any_unpriced else ""
        est_str = f" (~${cost:.2f} est.{note})"
    elif any_unpriced:
        est_str = " (cost n/a — set input_price/output_price)"
    else:
        est_str = ""
    if cap > 0:
        remaining = max(0, cap - total)
        head = (f"spend ledger: {total:,} / {cap:,} tokens today (UTC) — "
                f"{remaining:,} remaining{est_str}")
    else:
        head = f"spend ledger: {total:,} tokens today (UTC) — cap disabled{est_str}"
    lines = [head]
    for model, (p, c) in sorted(by_model.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        toks = p + c
        lines.append(f"  {model:<16}{toks:>12,} tokens   "
                     f"({_humanize(p)} prompt / {_humanize(c)} completion)")
    return "\n".join(lines)


def _render_once() -> str:
    ledger = get_shared_ledger()
    cost, any_unpriced = ledger.persisted_today_cost()
    return format_status(
        ledger.persisted_today_by_model(),
        ledger.persisted_today_tokens(),
        default_daily_cap(),
        cost,
        any_unpriced,
    )


def cmd_status(args) -> int:
    interval = getattr(args, "watch", None)
    if interval is None:
        print(_render_once())
        return 0
    try:
        while True:
            print("\033[2J\033[H", end="")
            print(_render_once())
            time.sleep(float(interval))
    except KeyboardInterrupt:
        return 0


def cmd_path(args) -> int:
    print(resolve_ledger_path())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="otm",
        description="Monitor OpenAI token usage against a daily (UTC) cap.",
    )
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="today's usage vs. cap (the monitor snapshot)")
    p_status.add_argument(
        "--watch", nargs="?", const=2.0, type=float, default=None, metavar="SECONDS",
        help="re-print every SECONDS (default 2) until Ctrl-C",
    )
    p_status.set_defaults(func=cmd_status)

    p_path = sub.add_parser("path", help="print the resolved ledger file path")
    p_path.set_defaults(func=cmd_path)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return func(args)
