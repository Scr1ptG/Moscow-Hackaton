"""Фоновые циклы backend: приём телеметрии, пересылка в ML, прогнозы, алерты, метрики.

Надёжность:

* ML недоступен -> телеметрия копится в ограниченном буфере и досылается позже,
  прогнозы помечаются устаревшими (показывается последнее известное состояние);
* поток телеметрии оборвался -> ТС помечаются «нет связи», сервис продолжает работать.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import numpy as np

from .alerts import update_alerts
from .config import Settings
from .ml_client import MLClient, MLUnavailable
from .schedule import Schedule
from .state import Store, Vehicle

log = logging.getLogger("backend.worker")


class Worker:
    """Оркестратор: связывает поток телеметрии, ML-сервис и дашборд."""

    def __init__(self, store: Store, schedule: Schedule, ml: MLClient, cfg: Settings,
                 broadcast: Callable[[dict], Awaitable[None]] | None = None):
        self.store, self.schedule, self.ml, self.cfg = store, schedule, ml, cfg
        self.broadcast = broadcast
        self.ml_synced_t = 0.0  # время последней телеметрии, уже доставленной в ML
        self._tasks: list[asyncio.Task] = []
        self._scheduled = set(schedule.vehicle_ids())

    # ------------------------------------------------------------ приём
    def ingest(self, records: list[dict]) -> int:
        """Принимает батч навигационных записей (из NDTP-сервера)."""
        s = self.store
        n = 0
        for r in records:
            tr = r.get("tr_id")
            if tr is None and r.get("unit_id") is not None:
                tr = self.schedule.unit_to_tr.get(int(r["unit_id"]))
            if tr is None:
                s.dropped += 1
                continue
            tr = int(tr)
            v = s.vehicles.get(tr) or s.vehicles.setdefault(tr, Vehicle(tr_id=tr, unit_id=r.get("unit_id")))
            t = float(r["t"])
            v.packets += 1
            if v.last_t is None or t >= v.last_t:
                v.last_t = t
                v.speed, v.heading = r.get("speed"), r.get("heading")
                if r.get("valid") and r.get("lon") is not None:
                    v.lon, v.lat, v.valid, v.last_valid_t = r["lon"], r["lat"], True, t
                else:
                    v.valid = False
            if len(s.buffer) >= self.cfg.max_buffer:
                s.buffer.popleft()
                s.dropped += 1
            s.buffer.append({**r, "tr_id": tr})
            s.stream_now = max(s.stream_now, t)
            n += 1
        s.received += n
        return n

    # ------------------------------------------------------------ циклы
    def start(self):
        self._tasks = [asyncio.create_task(c()) for c in (self._schedule_loop, self._forward_loop, self._predict_loop)]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _schedule_loop(self):
        """Синхронизирует расписание с ML-сервисом (повторяет, пока ML не поднимется)."""
        while not self.store.ml_schedule_loaded:
            try:
                await self.ml.health()  # дождаться, пока ML-сервис поднимется
                await self.ml.load_schedule(self.cfg.schedule_split, self.schedule.unit_to_tr)
                self.store.ml_schedule_loaded = True
                log.info("расписание загружено в ML-сервис")
            except MLUnavailable as e:
                log.warning("ML недоступен для загрузки расписания: %s", e)
                await asyncio.sleep(3)

    async def _forward_loop(self):
        while True:
            await asyncio.sleep(self.cfg.forward_every_s)
            await self.forward_once()

    async def forward_once(self, max_batch: int = 5000) -> int:
        s = self.store
        if not s.buffer or not s.ml_schedule_loaded:
            return 0
        batch = [s.buffer.popleft() for _ in range(min(len(s.buffer), max_batch))]
        try:
            await self.ml.send_telemetry(batch)
        except MLUnavailable:
            s.buffer.extendleft(reversed(batch))  # вернуть и дослать позже
            s.ml_degraded = True
            return 0
        self.ml_synced_t = max(self.ml_synced_t, max(r["t"] for r in batch))
        return len(batch)

    async def _predict_loop(self):
        while True:
            await asyncio.sleep(self.cfg.predict_every_s)
            await self.predict_once()

    async def predict_once(self) -> list[dict]:
        """Один цикл: прогнозы на «сейчас» потока -> инциденты -> метрики -> рассылка."""
        s = self.store
        now = self.ml_synced_t
        if not s.ml_schedule_loaded or now <= s.last_cycle_now:
            return []
        try:
            res = await self.ml.predictions(now)
        except MLUnavailable as e:
            s.ml_degraded = True
            for p in s.predictions.values():
                p["stale"] = True  # показываем последнее известное состояние
            await self._emit({"type": "degraded", "reason": str(e), "now": now})
            return []
        s.ml_degraded = False
        s.cycles += 1
        s.last_cycle_now, s.last_cycle_wall = now, time.time()
        items = res.get("items", [])
        for it in items:
            it["stale"] = False
            tr = int(it["tr_id"])
            s.predictions[tr] = it
            s.history(tr).append({"t": now, "delay_pred": it.get("delay_pred"), "risk": it.get("risk"), "target_stop_id": it.get("target_stop_id")})
            self._register_eval(it, now)
        self._score_eval(now)
        events = update_alerts(s, self.schedule, items, now, self.cfg)
        for e in events:
            await self._emit(e)
        await self._emit({"type": "cycle", "now": now, "vehicles": self.vehicles_snapshot(), "ml_latency_ms": res.get("latency_ms")})
        return events

    # ------------------------------------------------------------ живая оценка качества
    def _register_eval(self, it: dict, now: float):
        if not self.schedule.has_facts:
            return
        stop = int(it["target_stop_id"])
        fact = self.schedule.fact_of(stop)
        if fact is None:
            return
        key = (int(it["tr_id"]), stop, int(now // 60))
        self.store.pending_eval[key] = (float(it["delay_pred"]), float(it.get("cur_dev_s") or 0), fact, float(it["target_plan"]))

    def _score_eval(self, now: float):
        done = [k for k, v in self.store.pending_eval.items() if v[2] <= now]
        for k in done:
            pred, cur, fact, plan = self.store.pending_eval.pop(k)
            dev = fact - plan
            self.store.scored.append((dev - pred, dev - cur, dev))

    # ------------------------------------------------------------ представления
    def vehicles_snapshot(self) -> list[dict]:
        s = self.store
        out = []
        for tr, v in s.vehicles.items():
            p = s.predictions.get(tr, {})
            stale = v.last_t is None or s.stream_now - v.last_t > self.cfg.stale_after_s
            out.append({
                "tr_id": tr, "scheduled": tr in self._scheduled, "lon": v.lon, "lat": v.lat, "speed": v.speed, "heading": v.heading,
                "last_t": v.last_t, "gps_valid": v.valid, "connection": "lost" if stale else "ok",
                "risk": p.get("risk", "unknown"), "delay_pred": p.get("delay_pred"), "p_late": p.get("p_late"),
                "cause": p.get("cause"), "target_stop_id": p.get("target_stop_id"),
                "prediction_stale": bool(p.get("stale", True)) if p else True,
                "alert_id": s.open_alert_by_tr.get(tr),
            })
        return out

    async def _emit(self, event: dict):
        if self.broadcast:
            try:
                await self.broadcast(event)
            except Exception as e:  # noqa: BLE001 — рассылка не должна ронять цикл
                log.warning("broadcast failed: %s", e)


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values)
    return {"p50_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1), "n": int(len(a))}
