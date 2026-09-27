"""Асинхронный клиент ML-сервиса с circuit breaker.

Если ML-сервис N раз подряд не ответил, breaker «размыкается»: запросы не шлются
``cooldown`` секунд (backend работает на последнем известном состоянии), затем
пробный запрос (half-open) — при успехе breaker замыкается.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx


class MLUnavailable(RuntimeError):
    """ML-сервис недоступен (ошибка запроса или разомкнут breaker)."""


@dataclass
class CircuitBreaker:
    failures_to_open: int = 3
    cooldown_s: float = 10.0
    failures: int = 0
    opened_at: float | None = None
    total_failures: int = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if time.monotonic() - self.opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self):
        self.failures, self.opened_at = 0, None

    def failure(self):
        self.failures += 1
        self.total_failures += 1
        if self.failures >= self.failures_to_open or self.state == "half-open":
            self.opened_at = time.monotonic()


@dataclass
class MLClient:
    """Клиент ML-сервиса (контракт — ``services/ml_service/app.py``)."""

    base_url: str
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    latencies_ms: list = field(default_factory=list)

    def __post_init__(self):
        self._http = httpx.AsyncClient(base_url=self.base_url.rstrip("/"), timeout=self.timeout_s)

    async def close(self):
        await self._http.aclose()

    async def _call(self, method: str, path: str, **kw):
        if not self.breaker.allow():
            raise MLUnavailable("circuit breaker open")
        t0 = time.perf_counter()
        try:
            r = await self._http.request(method, path, **kw)
            r.raise_for_status()
        except (httpx.HTTPError, OSError) as e:
            self.breaker.failure()
            raise MLUnavailable(str(e)) from e
        self.breaker.success()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        del self.latencies_ms[:-500]
        return r.json()

    async def health(self) -> dict:
        return await self._call("GET", "/health")

    async def load_schedule(self, split: str, unit_to_tr: dict[int, int]) -> dict:
        # загрузка расписания в ML на холодном старте может занять десятки секунд
        return await self._call("POST", "/v1/schedule", timeout=90,
                                json={"split": split, "unit_to_tr": {str(k): v for k, v in unit_to_tr.items()}})

    async def send_telemetry(self, records: list[dict]) -> dict:
        return await self._call("POST", "/v1/telemetry", json=records)

    async def predictions(self, now: float) -> dict:
        return await self._call("GET", "/v1/predictions", params={"now": now})

    async def whatif(self, payload: dict) -> dict:
        return await self._call("POST", "/v1/whatif", json=payload)

    async def model_info(self) -> dict:
        return await self._call("GET", "/v1/model")

    async def reload_model(self) -> dict:
        return await self._call("POST", "/v1/admin/reload")
