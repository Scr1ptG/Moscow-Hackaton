"""Настройки backend (переменные окружения)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Параметры сервиса; все задаются через окружение (см. README)."""

    ml_url: str = field(default_factory=lambda: _env("ML_URL", "http://localhost:8001"))
    data_dir: str = field(default_factory=lambda: _env("MT_DATA_DIR", r"C:\Users\ibrag\Downloads\dataset"))
    #: какой день обслуживаем (для демо — день раздачи: test/validate/train)
    schedule_split: str = field(default_factory=lambda: _env("SCHEDULE_SPLIT", "test"))
    #: как часто (в секундах реального времени) запрашивать прогнозы
    predict_every_s: float = field(default_factory=lambda: float(_env("PREDICT_EVERY_S", "2")))
    #: как часто пересылать накопленную телеметрию в ML
    forward_every_s: float = field(default_factory=lambda: float(_env("FORWARD_EVERY_S", "0.5")))
    ml_timeout_s: float = field(default_factory=lambda: float(_env("ML_TIMEOUT_S", "10")))
    #: circuit breaker: после N ошибок подряд ML считается недоступным на cooldown секунд
    breaker_failures: int = field(default_factory=lambda: int(_env("BREAKER_FAILURES", "3")))
    breaker_cooldown_s: float = field(default_factory=lambda: float(_env("BREAKER_COOLDOWN_S", "10")))
    #: ТС без телеметрии дольше этого (время потока) — «связь потеряна»
    stale_after_s: float = field(default_factory=lambda: float(_env("STALE_AFTER_S", "300")))
    #: минимальный уровень риска для алерта: yellow | red
    alert_min_risk: str = field(default_factory=lambda: _env("ALERT_MIN_RISK", "yellow"))
    #: сколько «зелёных» циклов подряд нужно для закрытия алерта
    resolve_after_cycles: int = field(default_factory=lambda: int(_env("RESOLVE_AFTER_CYCLES", "2")))
    #: буфер телеметрии на время недоступности ML (записей)
    max_buffer: int = field(default_factory=lambda: int(_env("MAX_BUFFER", "300000")))
    ndtp_host: str = field(default_factory=lambda: _env("NDTP_HOST", "localhost"))
    ndtp_port: int = field(default_factory=lambda: int(_env("NDTP_PORT", "9201")))
    emulator_url: str = field(default_factory=lambda: _env("EMULATOR_URL", "http://localhost:18080"))
    #: куда эмулятор шлёт NDTP: из Docker-сети — "ingest", с хоста — "host.docker.internal"
    emulator_target_host: str = field(default_factory=lambda: _env("EMULATOR_TARGET_HOST", "host.docker.internal"))
