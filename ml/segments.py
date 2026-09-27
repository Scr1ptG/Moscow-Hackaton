"""Проблемные участки: где ТС сегодня систематически теряют время (по фактическим прибытиям из GPS).

Для каждого перегона «остановка k -> k+1» внутри рейса считается прирост отклонения
``(arr_{k+1} - plan_{k+1}) - (arr_k - plan_k)`` по прибытиям, подтверждённым к моменту
``now`` (то есть без заглядывания в будущее). Перегоны агрегируются по месту остановок
(один и тот же участок в разных рейсах), результат — средняя потеря времени и число проездов.
Старт рейса исключается: «прибытие» на первую остановку рейса — шум (ТС стоит на конечной).
"""
from __future__ import annotations

import numpy as np

from .features import Context, arrivals_known_by


def problem_segments(ctx: Context, now: float, min_obs: int = 2, top: int = 50) -> list[dict]:
    """Список перегонов со средней потерей времени, по убыванию потери."""
    agg: dict[tuple, dict] = {}
    for tr, st in ctx.schedule.items():
        arr = arrivals_known_by(st, now, "arr")
        dev = arr - st["plan"].to_numpy()
        trip = st["trip_no"].to_numpy()
        tstart = st["is_trip_start"].to_numpy()
        lk = st["loc_key"].to_numpy()
        lon, lat, sid = st["lon"].to_numpy(), st["lat"].to_numpy(), st["stop_id"].to_numpy()
        for k in range(len(st) - 1):
            if trip[k] != trip[k + 1] or tstart[k] or np.isnan(dev[k]) or np.isnan(dev[k + 1]):
                continue
            key = (int(lk[k]), int(lk[k + 1]))
            a = agg.setdefault(key, {"gains": [], "tr_ids": set(), "last_t": 0.0,
                                     "from": [float(lon[k]), float(lat[k])], "to": [float(lon[k + 1]), float(lat[k + 1])],
                                     "from_stop_id": int(sid[k]), "to_stop_id": int(sid[k + 1])})
            a["gains"].append(float(dev[k + 1] - dev[k]))
            a["tr_ids"].add(int(tr))
            a["last_t"] = max(a["last_t"], float(arr[k + 1]))
    out = []
    for a in agg.values():
        g = np.asarray(a["gains"])
        if len(g) < min_obs:
            continue
        out.append({
            "from": a["from"], "to": a["to"], "from_stop_id": a["from_stop_id"], "to_stop_id": a["to_stop_id"],
            "mean_loss_s": round(float(g.mean()), 1), "median_loss_s": round(float(np.median(g)), 1),
            "n": int(len(g)), "tr_ids": sorted(a["tr_ids"]), "last_t": a["last_t"],
        })
    out.sort(key=lambda s: -s["mean_loss_s"])
    return out[:top]
