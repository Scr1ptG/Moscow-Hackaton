"""Треки ТС и детекция прибытий на остановки по GPS (облегчённый map matching).

Идея: плановое расписание задаёт последовательность остановок с координатами.
Трек ТС (валидные GPS-точки) интерполируется линейно между соседними
фиксациями, и для каждой остановки ищется момент **первого входа** в круг
радиуса ``R`` вокруг неё в окне ``[plan - before, plan + after]``.

Момент первого входа — причинная величина: если вход случился до ``T``, то он
одинаков и на полном треке, и на треке, обрезанном по ``T``. Поэтому прибытия
можно посчитать один раз на весь трек, а при построении признаков для точки
``T`` брать только прибытия с ``arrival <= T`` — утечки нет.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import to_xy

#: Максимальный разрыв между фиксациями, через который ещё интерполируем трек, с.
MAX_INTERP_GAP_S = 150.0


@dataclass
class Track:
    """Трек одного ТС: все пакеты (для статистики связи) и валидные точки."""

    tr_id: int
    t_all: np.ndarray  # времена всех пакетов
    valid_all: np.ndarray  # флаг валидности координат у всех пакетов
    speed_all: np.ndarray  # скорость (NaN, если нет)
    t: np.ndarray  # времена валидных точек
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    heading: np.ndarray


def build_tracks(traffic: pd.DataFrame) -> dict[int, Track]:
    """Группирует телеметрию по ТС в :class:`Track` (данные уже отсортированы)."""
    tracks: dict[int, Track] = {}
    for tr, g in traffic.groupby("tr_id", sort=False):
        v = g[g["valid"]]
        x, y = to_xy(v["lon"].to_numpy(), v["lat"].to_numpy())
        tracks[int(tr)] = Track(
            tr_id=int(tr),
            t_all=g["t"].to_numpy(),
            valid_all=g["valid"].to_numpy(),
            speed_all=g["speed"].to_numpy(),
            t=v["t"].to_numpy(),
            x=x,
            y=y,
            lon=v["lon"].to_numpy(),
            lat=v["lat"].to_numpy(),
            speed=v["speed"].to_numpy(),
            heading=v["heading"].to_numpy(),
        )
    return tracks


def _first_entry(tr: Track, sx: float, sy: float, t0: float, t1: float, R: float) -> tuple[float, float, float]:
    """Первый вход трека в круг (sx, sy, R) на интервале [t0, t1].

    Возвращает ``(время_входа, время_подтверждения, мин_дистанция_в_окне)``.
    Время входа интерполируется между фиксациями (движение линейное, если
    разрыв не больше :data:`MAX_INTERP_GAP_S`), поэтому о входе становится
    известно только в момент *следующей* фиксации — это «время подтверждения».
    Признаки на момент ``T`` обязаны фильтровать прибытия по нему, иначе в
    них протекает до одного интервала GPS (~15 с) будущего.
    """
    i0 = np.searchsorted(tr.t, t0, side="left")
    i1 = np.searchsorted(tr.t, t1, side="right")
    if i1 - i0 < 1:
        return np.nan, np.nan, np.nan
    t = tr.t[i0:i1]
    ax, ay = tr.x[i0:i1] - sx, tr.y[i0:i1] - sy
    d = np.hypot(ax, ay)
    dmin = float(d.min())
    if d[0] <= R:
        return float(t[0]), float(t[0]), dmin
    if len(t) < 2:
        return np.nan, np.nan, dmin
    # отрезки между соседними фиксациями
    dx, dy = np.diff(ax), np.diff(ay)
    dt = np.diff(t)
    A = dx * dx + dy * dy
    B = 2 * (ax[:-1] * dx + ay[:-1] * dy)
    C = ax[:-1] ** 2 + ay[:-1] ** 2 - R * R
    disc = B * B - 4 * A * C
    ok = (A > 0) & (disc >= 0) & (dt <= MAX_INTERP_GAP_S)
    u = np.full(len(A), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = (-B - np.sqrt(np.where(ok, disc, 0))) / (2 * np.where(A > 0, A, 1))
    hit = ok & (u1 >= 0) & (u1 <= 1)
    u[hit] = u1[hit]
    # вход в точке-фиксации (если интерполяция невозможна)
    pt_hit = d[1:] <= R
    seg_t = t[:-1] + np.where(hit, u, 0.0) * dt
    cand = np.minimum(np.where(hit, seg_t, np.inf), np.where(pt_hit, t[1:], np.inf))
    if not np.isfinite(cand).any():
        return np.nan, np.nan, dmin
    k = int(np.argmin(cand))  # отрезок k -> k+1; подтверждение приходит с фиксацией k+1
    return float(cand[k]), float(t[k + 1]), dmin


def first_entries(
    tr: Track | None,
    plan: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    R: float,
    before_s: float = 5 * 60,
    after_s: float = 14 * 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy-ядро детектора: ``(arr, arr_known, dmin)`` для остановок (без pandas)."""
    n = len(plan)
    arr, known, dmin = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if tr is not None and len(tr.t):
        for i in range(n):
            arr[i], known[i], dmin[i] = _first_entry(tr, x[i], y[i], plan[i] - before_s, plan[i] + after_s, R)
    return arr, known, dmin


def detect_arrivals(
    tr: Track | None,
    stops: pd.DataFrame,
    R: float = 40.0,
    before_s: float = 5 * 60,
    after_s: float = 14 * 60,
    monotone: bool = True,
) -> pd.DataFrame:
    """Детектирует прибытия ТС на остановки его плана.

    :param tr: трек ТС (``None`` — телеметрии нет).
    :param stops: плановые остановки ТС (``stop_id, plan, x, y``), отсортированы.
    :param R: радиус зоны остановки, м.
    :param before_s: насколько раньше плана может прийти ТС.
    :param after_s: насколько позже плана может прийти ТС.
    :param monotone: отбрасывать прибытия, нарушающие порядок остановок.
        Фильтр смотрит на все прибытия сразу, т.е. **не причинный** — для
        признаков используйте ``monotone=False`` и причинный фильтр в
        :mod:`ml.features`.
    :returns: ``stops`` с колонками ``arr`` (время входа или NaN), ``arr_known``
        (время, когда вход стал известен по GPS) и ``dmin``.
    """
    arr, known, dmin = first_entries(
        tr, stops["plan"].to_numpy(), stops["x"].to_numpy(), stops["y"].to_numpy(), R, before_s, after_s
    )
    out = stops.copy()
    out["arr"] = arr
    out["arr_known"] = known
    out["dmin"] = dmin
    if not monotone:
        return out
    # прибытия должны идти в порядке остановок: отбрасываем «прыжки назад»
    a = out["arr"].to_numpy(copy=True)
    run_max = -np.inf
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        if a[i] + 30 < run_max:  # небольшой допуск на совпадающие по времени остановки
            a[i] = np.nan
        else:
            run_max = max(run_max, a[i])
    out["arr"] = a
    return out
