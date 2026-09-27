"""REST API и WebSocket для дашборда диспетчера (OpenAPI/Swagger: ``/docs``)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .alerts import alert_card, alert_summary
from .ml_client import MLUnavailable
from .worker import latency_stats

router = APIRouter(prefix="/api/v1")
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ схемы
class NavRecordIn(BaseModel):
    """Навигационная запись (ячейка NDTP ``G6CellNav00`` после парсинга)."""

    unit_id: Optional[int] = Field(None, description="ID бортового терминала (peerAddress)")
    tr_id: Optional[int] = Field(None, description="ID ТС, если известен")
    t: float = Field(..., description="Unix-время фиксации, с")
    valid: bool = True
    lon: Optional[float] = None
    lat: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None


class VehicleOut(BaseModel):
    tr_id: int
    scheduled: bool
    lon: Optional[float] = None
    lat: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    last_t: Optional[float] = None
    gps_valid: bool
    connection: str = Field(..., description="ok | lost (нет телеметрии дольше STALE_AFTER_S)")
    risk: str = Field(..., description="green | yellow | red | unknown")
    delay_pred: Optional[float] = Field(None, description="прогноз отклонения на цели через 10–15 мин, с")
    p_late: Optional[float] = None
    cause: Optional[str] = None
    target_stop_id: Optional[int] = None
    prediction_stale: bool
    alert_id: Optional[str] = None


class AlertOut(BaseModel):
    id: str
    tr_id: int
    status: str
    severity: str
    opened_at: float
    updated_at: float
    resolved_at: Optional[float] = None
    delay_pred: Optional[float] = None
    p_late: Optional[float] = None
    cause: Optional[str] = None
    target_address: Optional[str] = None
    peak_delay_s: float


class WhatIfIn(BaseModel):
    """Сценарий диспетчера для ТС (к его последнему прогнозу)."""

    tr_id: int
    delay_shift_s: float = Field(0, description="ТС опоздает ещё на столько секунд (+) / нагонит (−)")
    extra_layover_min: float = Field(0, description="добавить минут отстоя на конечной (если отстой есть на пути)")
    reserve_vehicle: bool = Field(False, description="выпустить резервное ТС по графику")
    overrides: dict[str, float] = Field(default_factory=dict, description="прямое переопределение признаков")


class ReplayIn(BaseModel):
    split: str = "test"
    speed: float = 60.0
    start: Optional[str] = None
    end: Optional[str] = None


class EmulatorIn(BaseModel):
    units: list[int] = Field(..., description="unitId эмулируемых терминалов")
    interval_ms: int = 5000
    target_host: Optional[str] = Field(None, description="по умолчанию EMULATOR_TARGET_HOST")
    target_port: int = 9201


# ------------------------------------------------------------------ helpers
def _w(request: Request):
    return request.app.state.worker


# ------------------------------------------------------------------ служебные
@router.get("/health", summary="Состояние сервиса и зависимостей")
async def health(request: Request):
    w = _w(request)
    s = w.store
    return {
        "status": "degraded" if s.ml_degraded else "ok",
        "ml": {"breaker": w.ml.breaker.state, "schedule_loaded": s.ml_schedule_loaded, "failures": w.ml.breaker.total_failures},
        "stream_now": s.stream_now, "ml_synced_t": w.ml_synced_t, "buffered": len(s.buffer),
        "received": s.received, "dropped": s.dropped, "cycles": s.cycles,
        "vehicles": len(s.vehicles), "open_alerts": len(s.open_alert_by_tr),
    }


@router.post("/telemetry", summary="Приём батча телеметрии от NDTP-сервера")
async def telemetry(request: Request, records: list[NavRecordIn]):
    n = _w(request).ingest([r.model_dump() for r in records])
    return {"accepted": n}


# ------------------------------------------------------------------ дашборд
@router.get("/vehicles", response_model=list[VehicleOut], summary="ТС на карте: положение, риск, прогноз")
async def vehicles(request: Request):
    return _w(request).vehicles_snapshot()


@router.get("/vehicles/{tr_id}", summary="Карточка ТС: состояние, последний прогноз, история прогнозов")
async def vehicle(request: Request, tr_id: int):
    w = _w(request)
    v = next((x for x in w.vehicles_snapshot() if x["tr_id"] == tr_id), None)
    if v is None:
        raise HTTPException(404, "ТС не найдено")
    return {**v, "prediction": w.store.predictions.get(tr_id), "history": list(w.store.history(tr_id))[-120:],
            "route": w.schedule.route_line(tr_id)}


@router.get("/routes", summary="Линии маршрутов для карты")
async def routes(request: Request):
    sch = _w(request).schedule
    return [{"tr_id": tr, "line": sch.route_line(tr)} for tr in sch.vehicle_ids()]


@router.get("/alerts", response_model=list[AlertOut], summary="Инциденты (по умолчанию активные)")
async def alerts(request: Request, status: str = "active"):
    s = _w(request).store
    items = [a for a in s.alerts.values() if status == "all" or (status == "active" and a.status != "resolved") or a.status == status]
    items.sort(key=lambda a: (a.severity != "red", -a.updated_at))
    return [alert_summary(a) for a in items]


@router.get("/alerts/{alert_id}", summary="Карточка инцидента: прогноз, интервал, причина, объяснение, участок")
async def alert(request: Request, alert_id: str):
    a = _w(request).store.alerts.get(alert_id)
    if a is None:
        raise HTTPException(404, "инцидент не найден")
    return alert_card(a)


@router.post("/alerts/{alert_id}/ack", summary="Диспетчер принял инцидент в работу")
async def ack(request: Request, alert_id: str):
    a = _w(request).store.alerts.get(alert_id)
    if a is None:
        raise HTTPException(404, "инцидент не найден")
    if a.status == "open":
        a.status = "acknowledged"
    return alert_summary(a)


@router.post("/whatif", summary="What-if: как изменится прогноз при действии диспетчера")
async def whatif(request: Request, body: WhatIfIn):
    w = _w(request)
    p = w.store.predictions.get(body.tr_id)
    if p is None:
        raise HTTPException(404, "по ТС ещё нет прогноза")
    overrides, shifts = dict(body.overrides), {}
    if body.delay_shift_s:
        shifts.update(cur_dev_s=body.delay_shift_s, det_plus_hist_gain=body.delay_shift_s)
    if body.extra_layover_min:
        shifts["layover_slack_s"] = body.extra_layover_min * 60
    if body.reserve_vehicle:
        overrides.update(cur_dev_s=0.0, det_plus_hist_gain=0.0, hist_tgt_delay=0.0)
    payload = {"tr_id": body.tr_id, "T": p["T"], "target_stop_id": p["target_stop_id"], "target_plan": p["target_plan"],
               "cur_dev_s": p.get("cur_dev_s", 0.0), "overrides": overrides, "shifts": shifts}
    try:
        return await w.ml.whatif(payload)
    except MLUnavailable as e:
        raise HTTPException(503, f"ML-сервис недоступен: {e}")


@router.get("/metrics", summary="Качество онлайн (на реплее дня с фактом) и производительность")
async def metrics(request: Request):
    w = _w(request)
    s = w.store
    return {
        "quality_live": {**s.quality(), "note": "факт используется только для оценки, в ML не передаётся"},
        "ml_latency": latency_stats(w.ml.latencies_ms),
        "cycles": s.cycles, "received": s.received, "buffered": len(s.buffer), "dropped": s.dropped,
        "stream_lag_s": round(s.stream_now - w.ml_synced_t, 1),
    }


@router.get("/model", summary="Карточка модели (из ML-сервиса)")
async def model(request: Request):
    try:
        return await _w(request).ml.model_info()
    except MLUnavailable as e:
        raise HTTPException(503, f"ML-сервис недоступен: {e}")


# ------------------------------------------------------------------ администрирование
@router.post("/admin/reload-model", summary="Горячая перезагрузка модели в ML-сервисе")
async def reload_model(request: Request):
    try:
        return await _w(request).ml.reload_model()
    except MLUnavailable as e:
        raise HTTPException(503, str(e))


@router.post("/admin/replay", summary="Запустить реплей исторического дня в NDTP (демо)")
async def replay(request: Request, body: ReplayIn):
    cfg = request.app.state.cfg
    old = getattr(request.app.state, "replay_proc", None)
    if old is not None and old.poll() is None:
        old.terminate()
    cmd = [sys.executable, "-m", "ndtp.replay", "--split", body.split, "--host", cfg.ndtp_host,
           "--port", str(cfg.ndtp_port), "--speed", str(body.speed)]
    if body.start:
        cmd += ["--start", body.start]
    if body.end:
        cmd += ["--end", body.end]
    request.app.state.replay_proc = subprocess.Popen(cmd, cwd=ROOT)
    return {"started": True, "pid": request.app.state.replay_proc.pid, "cmd": " ".join(cmd[1:])}


@router.post("/admin/emulator", summary="Настроить эмулятор NDTP (проксирует POST /api/config)")
async def emulator(request: Request, body: EmulatorIn):
    cfg = request.app.state.cfg
    conf = {"targetHost": body.target_host or cfg.emulator_target_host, "targetPort": body.target_port,
            "units": [{"unitId": u, "intervalMs": body.interval_ms, "autoGenerate": True, "cells": []} for u in body.units]}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(cfg.emulator_url.rstrip("/") + "/api/config", json=conf)
            return {"status": r.status_code, "response": r.text[:500], "config": conf}
    except httpx.HTTPError as e:
        raise HTTPException(503, f"эмулятор недоступен: {e}")


# ------------------------------------------------------------------ WebSocket
@router.websocket("/ws")
async def ws(websocket: WebSocket):
    """Поток событий: ``cycle`` (снимок ТС), ``alert_opened|updated|resolved``, ``degraded``."""
    hub = websocket.app.state.hub
    await websocket.accept()
    hub.add(websocket)
    try:
        await websocket.send_json({"type": "hello", "vehicles": websocket.app.state.worker.vehicles_snapshot()})
        while True:
            await websocket.receive_text()  # клиент может слать ping
    except WebSocketDisconnect:
        pass
    finally:
        hub.discard(websocket)
