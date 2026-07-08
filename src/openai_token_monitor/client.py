"""Drop-in metered OpenAI clients (composition + __getattr__ delegation)."""
from __future__ import annotations

import openai

from openai_token_monitor import ledger as ledger_mod
from openai_token_monitor.config import default_daily_cap
from openai_token_monitor.ledger import SpendLedger, get_shared_ledger, register_atexit_flush


def _record_into(ledger, resp, kwargs, input_price=None, output_price=None) -> None:
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0)
        ledger.record(kwargs.get("model", ""), int(prompt or 0), int(completion or 0),
                      input_price, output_price)
        if ledger.unflushed_calls() >= ledger_mod.FLUSH_EVERY_N_CALLS:
            ledger.flush()
    except Exception as e:
        print(f"  ⚠ spend-meter accounting error (ignored, your call still succeeded): {e}")


def _resolve_ledger(ledger, ledger_path):
    if ledger is not None:
        return ledger
    if ledger_path is not None:
        led = SpendLedger(path=ledger_path)
        register_atexit_flush(led)
        return led
    return get_shared_ledger()


class _MeteredCompletions:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._ledger = ledger
        self._daily_cap = daily_cap
        self._real = real
        self._input_price = input_price
        self._output_price = output_price

    def create(self, **kwargs):
        self._ledger.check_cap(self._daily_cap)
        resp = self._real.create(**kwargs)
        _record_into(self._ledger, resp, kwargs, self._input_price, self._output_price)
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class _MeteredChat:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._real = real
        self.completions = _MeteredCompletions(ledger, daily_cap, real.completions,
                                               input_price, output_price)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _MeteredResponses:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._ledger = ledger
        self._daily_cap = daily_cap
        self._real = real
        self._input_price = input_price
        self._output_price = output_price

    def create(self, **kwargs):
        self._ledger.check_cap(self._daily_cap)
        resp = self._real.create(**kwargs)
        _record_into(self._ledger, resp, kwargs, self._input_price, self._output_price)
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class MeteredOpenAI:
    def __init__(self, *, daily_cap=None, ledger=None, ledger_path=None,
                 input_price=None, output_price=None, **openai_kwargs):
        self._ledger = _resolve_ledger(ledger, ledger_path)
        self._daily_cap = default_daily_cap() if daily_cap is None else daily_cap
        self._input_price = input_price
        self._output_price = output_price
        self._bind(openai.OpenAI(**openai_kwargs))

    def _bind(self, client) -> None:
        self._client = client
        self.chat = _MeteredChat(self._ledger, self._daily_cap, client.chat,
                                 self._input_price, self._output_price)
        if getattr(client, "responses", None) is not None:
            self.responses = _MeteredResponses(self._ledger, self._daily_cap, client.responses,
                                               self._input_price, self._output_price)

    @property
    def ledger(self):
        return self._ledger

    @property
    def daily_cap(self):
        return self._daily_cap

    @property
    def input_price(self):
        return self._input_price

    @property
    def output_price(self):
        return self._output_price

    def __getattr__(self, name):
        if name == "_client":
            raise AttributeError(name)
        return getattr(self._client, name)


class _MeteredAsyncCompletions:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._ledger = ledger
        self._daily_cap = daily_cap
        self._real = real
        self._input_price = input_price
        self._output_price = output_price

    async def create(self, **kwargs):
        self._ledger.check_cap(self._daily_cap)
        resp = await self._real.create(**kwargs)
        _record_into(self._ledger, resp, kwargs, self._input_price, self._output_price)
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class _MeteredAsyncChat:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._real = real
        self.completions = _MeteredAsyncCompletions(ledger, daily_cap, real.completions,
                                                    input_price, output_price)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _MeteredAsyncResponses:
    def __init__(self, ledger, daily_cap, real, input_price=None, output_price=None):
        self._ledger = ledger
        self._daily_cap = daily_cap
        self._real = real
        self._input_price = input_price
        self._output_price = output_price

    async def create(self, **kwargs):
        self._ledger.check_cap(self._daily_cap)
        resp = await self._real.create(**kwargs)
        _record_into(self._ledger, resp, kwargs, self._input_price, self._output_price)
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class MeteredAsyncOpenAI:
    def __init__(self, *, daily_cap=None, ledger=None, ledger_path=None,
                 input_price=None, output_price=None, **openai_kwargs):
        self._ledger = _resolve_ledger(ledger, ledger_path)
        self._daily_cap = default_daily_cap() if daily_cap is None else daily_cap
        self._input_price = input_price
        self._output_price = output_price
        self._bind(openai.AsyncOpenAI(**openai_kwargs))

    def _bind(self, client) -> None:
        self._client = client
        self.chat = _MeteredAsyncChat(self._ledger, self._daily_cap, client.chat,
                                      self._input_price, self._output_price)
        if getattr(client, "responses", None) is not None:
            self.responses = _MeteredAsyncResponses(self._ledger, self._daily_cap, client.responses,
                                                    self._input_price, self._output_price)

    @property
    def ledger(self):
        return self._ledger

    @property
    def daily_cap(self):
        return self._daily_cap

    @property
    def input_price(self):
        return self._input_price

    @property
    def output_price(self):
        return self._output_price

    def __getattr__(self, name):
        if name == "_client":
            raise AttributeError(name)
        return getattr(self._client, name)
