"""A tiny built-in price table, for convenience only. It holds ONLY the models whose
prices are verified against the original source; for anything else, pass input_price /
output_price (USD per 1M tokens) to MeteredOpenAI or SpendLedger.record and the estimate
comes straight from your numbers. This deliberately keeps the package out of the
price-maintenance business — see the README's "Custom pricing" section.
"""
from __future__ import annotations

# (prompt_price_per_1M_usd, completion_price_per_1M_usd) — verified against the KTH-Agent
# source. Deliberately minimal: supply your own prices for other models rather than
# trusting a guessed table.
OPENAI_PRICE_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
}


def estimate_cost_usd(model, prompt_tokens, completion_tokens,
                      input_price=None, output_price=None):
    """USD estimate. Per side, use the supplied price if given, else the built-in table's
    price for `model`, else that side is unresolved. Returns None if either side is
    unresolved (tokens-only, never a half-price guess). Prices are USD per 1M tokens."""
    table = OPENAI_PRICE_PER_MILLION.get(model)
    in_price = input_price if input_price is not None else (table[0] if table else None)
    out_price = output_price if output_price is not None else (table[1] if table else None)
    if in_price is None or out_price is None:
        return None
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
