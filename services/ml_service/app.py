"""ML-сервис: онлайн-инференс прогноза задержек (FastAPI, OpenAPI/Swagger на /docs).

Контракт с backend:

1. ``POST /v1/schedule`` — загрузить плановое расписание дня (CSV-путь или строки).
2. ``POST /v1/telemetry`` — батч навигационных записей из NDTP (уже распарсенных).
3. ``GET  /v1/predictions?now=...`` — прогнозы для всех активных ТС на момент ``now``
   (цель — первая плановая остановка в окне (now+10, now+15] мин).
4. ``POST /v1/predict`` — прогноз для заданной точки (tr_id, T, цель, cur_dev_s).
5. ``POST /v1/whatif`` — пересчёт прогноза с переопределением признаков
   (например, «дать ТС дополнительные 3 мин отстоя»).

Запуск: ``uvicorn services.ml_service.app:app --port 8001``.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import load_schedule_plan, to_unix
from ml.online import OnlineState
from ml.predictor import load_predictor

@asynccontextmanager
async def _lifespan(app):
    """Прогрев: модель грузится при старте, чтобы первый запрос не упирался в таймауты клиентов."""
    try:
        _predictor()
    except Exception as e:  # noqa: BLE001 — сервис всё равно поднимается, /health покажет проблему
        print("[ml-service] модель не загружена:", e)
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="Transit Delay Predictor — ML service",
    version="1.0.0",
    description="Прогноз отклонения от графика на горизонте 10–15 мин по потоковой телеметрии NDTP.",
)

STATE: dict = {"online": None, "predictor": None, "latency_ms": []}


class NavRecordIn(BaseModel):
    """Навигационная запись (ячейка G6CellNav00 после парсинга NDTP)."""

    unit_id: Optional[int] = Field(None, description="ID бортового терминала (peerAddress)")
    tr_id: Optional[int] = Field(None, description="ID ТС, если известен")
    t: float = Field(..., description="Unix-время фиксации, с")
    valid: bool = True
    lon: Optional[float] = None
    lat: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None


class ScheduleIn(BaseModel):
    csv_path: Optional[str] = Field(None, description="Путь к schedule.csv / schedule_plan.csv")
    split: Optional[str] = Field(None, description="Или имя сплита датасета: train/test/validate")
    unit_to_tr: dict[int, int] = Field(default_factory=dict, description="Соответствие unit_id -> tr_id")


class PointIn(BaseModel):
    tr_id: int
    T: float = Field(..., description="Момент прогноза, Unix-с")
    target_stop_id: int
    target_plan: float = Field(..., description="Плановое время прибытия на целевую остановку, Unix-с")
    cur_dev_s: float = Field(0.0, description="Отклонение на последней пройденной остановке, с")


class WhatIfIn(PointIn):
    overrides: dict[str, float] = Field(default_factory=dict, description="Переопределения признаков, напр. {'cur_dev_s': 0}")
    shifts: dict[str, float] = Field(default_factory=dict, description="Сдвиги признаков, напр. {'layover_slack_s': 180} (+3 мин отстоя)")


def _predictor():
    """Продовая компактная модель (``ML_MODEL=compact``) или ансамбль (``ML_MODEL=ensemble``)."""
    if STATE["predictor"] is None:
        STATE["predictor"] = load_predictor()
    return STATE["predictor"]


def _shard() -> tuple[int, int] | None:
    """``ML_SHARD=i/N``: реплика обслуживает только ТС с ``ml.online.shard_of(tr_id, N) == i``."""
    import os

    v = os.environ.get("ML_SHARD")
    if not v:
        return None
    i, n = (int(x) for x in v.split("/"))
    return i, n


def _online() -> OnlineState:
    if STATE["online"] is None:
        raise HTTPException(409, "Сначала загрузите расписание: POST /v1/schedule")
    return STATE["online"]


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), 3)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out


@app.get("/health")
def health():
    """Живость сервиса и базовая статистика (для оркестратора)."""
    on = STATE["online"]
    lat = STATE["latency_ms"][-200:]
    return {
        "status": "ok",
        "schedule_loaded": on is not None,
        "vehicles_with_telemetry": on.vehicles_with_telemetry if on else 0,
        "last_telemetry_time": on.last_time if on else None,
        "p50_latency_ms": float(np.median(lat)) if lat else None,
    }


@app.get("/v1/model")
def model_info():
    """Карточка модели: признаки (с описанием и монотонностью), группы причин, метрики."""
    import json as _json
    from pathlib import Path

    p = _predictor()
    rep = Path(__file__).resolve().parents[2] / "reports"
    card = {"kind": p.meta.get("kind", "ensemble"), "features": p.feats}
    if card["kind"] == "compact":
        card.update(feature_info=p.info, groups={k: v["title"] for k, v in p.groups.items()}, postprocess=p.meta["postprocess"],
                    trained_at=p.meta.get("trained_at"))
    else:
        card.update(blend_weights=p.w, n_lgb=len(p.lgb), n_cat=len(p.cat), n_nn=len(p.nn))
    for name in ("cv_report_compact.json", "explainability.json"):
        f = rep / name
        if f.exists():
            d = _json.loads(f.read_text(encoding="utf-8"))
            card[name.replace(".json", "")] = {k: v for k, v in d.items() if k not in ("features", "importance_feature_mean_abs_s")}
    return card


@app.post("/v1/admin/reload")
def reload_model():
    """Горячая перезагрузка модели после продвижения новой версии (``python -m ml.retrain``)."""
    from ml import registry

    STATE["predictor"] = load_predictor()
    return {"reloaded": True, "version": registry.current(), "kind": STATE["predictor"].meta.get("kind")}


@app.post("/v1/schedule")
def load_schedule(body: ScheduleIn):
    """Загружает плановое расписание (только план, без факта)."""
    unit_to_tr = dict(body.unit_to_tr)
    if body.split:
        plan = load_schedule_plan(body.split)
        if not unit_to_tr:  # соответствие бортовой терминал -> ТС берём из телеметрии сплита
            from ml.data import DATA_DIR

            u = pd.read_csv(DATA_DIR / body.split / "traffic.csv", usecols=["unit_id", "tr_id"]).drop_duplicates()
            unit_to_tr = dict(zip(u["unit_id"].astype(int), u["tr_id"].astype(int)))
    elif body.csv_path:
        df = pd.read_csv(body.csv_path)
        from ml.data import parse_point_wkt

        lon, lat = parse_point_wkt(df["geom"])
        plan = pd.DataFrame({"stop_id": df["tt_action_item_id"], "tr_id": df["tr_id"], "plan": to_unix(df["time_begin"]), "lon": lon, "lat": lat})
    else:
        raise HTTPException(400, "Укажите csv_path или split")
    STATE["online"] = OnlineState(plan, unit_to_tr, wide=getattr(_predictor(), "needs_wide", True), shard=_shard())
    return {"vehicles": len(STATE["online"].schedule), "stops": int(len(plan)), "units_mapped": len(unit_to_tr)}


@app.post("/v1/telemetry")
def ingest(records: list[NavRecordIn]):
    """Принимает батч навигационных записей."""
    on = _online()
    for r in records:
        on.add_record(r.tr_id, r.t, r.valid, r.lon if r.lon is not None else np.nan, r.lat if r.lat is not None else np.nan,
                      r.speed if r.speed is not None else np.nan, r.heading if r.heading is not None else np.nan, r.unit_id)
    return {"accepted": len(records), "last_time": on.last_time}


@app.get("/v1/predictions")
def predictions(now: Optional[float] = None):
    """Прогнозы на горизонте 10–15 мин для всех ТС с телеметрией на момент ``now``."""
    on = _online()
    now = now or on.last_time
    t0 = time.perf_counter()
    ctx = on.context(now)
    pts = on.prediction_points(now, ctx)
    if pts.empty:
        return {"now": now, "items": []}
    from ml.features import point_features

    feats = pd.DataFrame([point_features(ctx, int(p.tr_id), now, int(p.target_stop_id), float(p.target_plan), float(p.cur_dev_s)) for p in pts.itertuples()])
    seq = None
    pr = _predictor()
    if pr.nn:
        from ml.nn import point_sequence

        seq = np.stack([point_sequence(ctx, int(p.tr_id), now, int(p.target_stop_id)) for p in pts.itertuples()])
    out = pr.predict_frame(feats, seq)
    items = []
    for i, p in enumerate(pts.itertuples()):
        trk = ctx.tracks.get(int(p.tr_id))
        pos = None
        if trk is not None and len(trk.t):
            pos = {"lon": float(trk.lon[-1]), "lat": float(trk.lat[-1]), "t": float(trk.t[-1]), "speed": float(trk.speed[-1]) if np.isfinite(trk.speed[-1]) else None}
        items.append({**_clean({k: v for k, v in p._asdict().items() if k != "Index"}), **_clean(out.iloc[i].to_dict()), "position": pos})
    dt = (time.perf_counter() - t0) * 1000
    STATE["latency_ms"].append(dt)
    return {"now": now, "latency_ms": round(dt, 1), "items": items}


@app.post("/v1/predict")
def predict(p: PointIn):
    """Прогноз задержки для одной прогнозной точки."""
    on = _online()
    ctx = on.context(p.T, [p.tr_id])
    t0 = time.perf_counter()
    r = _predictor().predict_point(ctx, p.tr_id, p.T, p.target_stop_id, p.target_plan, p.cur_dev_s)
    r.pop("features", None)
    STATE["latency_ms"].append((time.perf_counter() - t0) * 1000)
    return _clean(r)


@app.post("/v1/whatif")
def whatif(p: WhatIfIn):
    """What-if: прогноз до и после переопределения признаков (сценарий диспетчера)."""
    from ml.features import point_features

    on = _online()
    ctx = on.context(p.T, [p.tr_id])
    f = point_features(ctx, p.tr_id, p.T, p.target_stop_id, p.target_plan, p.cur_dev_s)
    g = {**f, **p.overrides}
    for k, v in p.shifts.items():  # сдвиг применим только к определённому признаку (NaN = неприменимо)
        if g.get(k) is not None and not (isinstance(g[k], float) and np.isnan(g[k])):
            g[k] = g[k] + v
    pr = _predictor()
    base = pr.predict_frame(pd.DataFrame([f])).iloc[0]
    new = pr.predict_frame(pd.DataFrame([g])).iloc[0]
    keep = ("delay_pred", "q10", "q90", "p_late", "risk", "cause", "explanation")
    return {
        "before": _clean({k: base[k] for k in keep if k in base}),
        "after": _clean({k: new[k] for k in keep if k in new}),
        "delta_s": round(float(new["delay_pred"] - base["delay_pred"]), 1),
    }
