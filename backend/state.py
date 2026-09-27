"""Состояние backend в памяти: ТС, прогнозы, инциденты, живые метрики.

Время в системе — «время потока» (метки телеметрии): при реплее исторического дня
с ускорением всё (алерты, устаревание данных, оценка качества) идёт по нему.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Vehicle:
    tr_id: int
    unit_id: int | None = None
    lon: float | None = None
    lat: float | None = None
    speed: float | None = None
    heading: float | None = None
    valid: bool = False
    last_t: float | None = None  # время последнего пакета (поток)
    last_valid_t: float | None = None
    packets: int = 0


@dataclass
class Alert:
    """Инцидент: ТС с высоким риском отклонения на горизонте 10–15 мин."""

    tr_id: int
    severity: str
    opened_at: float
    prediction: dict
    target: dict | None
    segment: list
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "open"  # open | acknowledged | resolved
    updated_at: float = 0.0
    resolved_at: float | None = None
    peak_delay_s: float = 0.0
    green_cycles: int = 0
    history: list = field(default_factory=list)  # (время, прогноз, риск)


@dataclass
class Store:
    vehicles: dict[int, Vehicle] = field(default_factory=dict)
    predictions: dict[int, dict] = field(default_factory=dict)  # последний прогноз по ТС
    pred_history: dict[int, deque] = field(default_factory=dict)
    alerts: dict[str, Alert] = field(default_factory=dict)
    open_alert_by_tr: dict[int, str] = field(default_factory=dict)
    buffer: deque = field(default_factory=deque)  # телеметрия к отправке в ML
    stream_now: float = 0.0
    last_cycle_now: float = 0.0
    last_cycle_wall: float | None = None
    cycles: int = 0
    received: int = 0
    dropped: int = 0
    ml_degraded: bool = False
    ml_schedule_loaded: bool = False
    # живая оценка качества (только при реплее дня, где известен факт)
    pending_eval: dict = field(default_factory=dict)  # (tr, stop, T_min) -> (pred, cur, fact_t, plan)
    scored: list = field(default_factory=list)  # (err_model, err_baseline)
    started_wall: float = field(default_factory=time.time)

    def history(self, tr_id: int) -> deque:
        return self.pred_history.setdefault(tr_id, deque(maxlen=240))

    def quality(self) -> dict:
        if not self.scored:
            return {"n": 0}
        a = np.asarray(self.scored, dtype=float)
        return {
            "n": int(len(a)),
            "mae_model_s": round(float(np.mean(np.abs(a[:, 0]))), 1),
            "mae_cur_dev_s": round(float(np.mean(np.abs(a[:, 1]))), 1),
            "mae_zero_s": round(float(np.mean(np.abs(a[:, 2]))), 1),
        }
