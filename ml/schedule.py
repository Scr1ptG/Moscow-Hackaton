"""Структура планового расписания ТС: порядок остановок, рейсы, отстой.

Ключевое наблюдение EDA: после отстоя на конечной (разрыв плана >= 5 мин,
остановка та же) задержка «обнуляется» — корреляция задержки до и после
отстоя ~0.03. Поэтому модели нужны признаки о том, есть ли отстой между
текущим положением ТС и целевой остановкой.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine, to_xy

#: Разрыв плана, начиная с которого считаем, что начинается новый рейс, с.
TRIP_GAP_S = 300.0
#: Остановка «та же», если ближе этого расстояния, м.
SAME_STOP_M = 150.0


def _order_ties(g: pd.DataFrame) -> pd.DataFrame:
    """Сортирует остановки по плану; при равном плане — жадно по близости.

    План дан с точностью до минуты, и в одну минуту часто попадают 2 остановки.
    Физически верный порядок — тот, что минимизирует путь, поэтому внутри
    группы с одинаковым временем берём ближайшую к предыдущей остановке.
    """
    g = g.sort_values(["plan", "stop_id"], kind="stable")
    plans = g["plan"].to_numpy()
    if len(np.unique(plans)) == len(plans):
        return g
    lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
    order: list[int] = []
    i = 0
    n = len(g)
    while i < n:
        j = i
        while j + 1 < n and plans[j + 1] == plans[i]:
            j += 1
        idx = list(range(i, j + 1))
        while idx:
            if order:
                p = order[-1]
                d = haversine(lon[p], lat[p], lon[idx], lat[idx])
                k = idx[int(np.nanargmin(d))] if np.isfinite(d).any() else idx[0]
            else:
                k = idx[0]
            order.append(k)
            idx.remove(k)
        i = j + 1
    return g.iloc[order]


def prepare_schedule(plan: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Готовит план по каждому ТС.

    Возвращает словарь ``tr_id -> DataFrame`` с колонками:
    ``stop_id, plan, lon, lat, x, y, gap_prev, dist_prev, cum_dist, trip_no,
    pos_in_trip, trip_len, is_trip_start, is_trip_end, loc_key``.
    """
    out: dict[int, pd.DataFrame] = {}
    for tr, g in plan.groupby("tr_id", sort=False):
        g = _order_ties(g).reset_index(drop=True)
        x, y = to_xy(g["lon"].to_numpy(), g["lat"].to_numpy())
        g["x"], g["y"] = x, y
        gap = np.diff(g["plan"].to_numpy(), prepend=np.nan)
        dist = np.r_[np.nan, haversine(g["lon"].to_numpy()[:-1], g["lat"].to_numpy()[:-1], g["lon"].to_numpy()[1:], g["lat"].to_numpy()[1:])]
        g["gap_prev"] = gap
        g["dist_prev"] = dist
        g["cum_dist"] = np.nan_to_num(dist).cumsum()
        start = np.isnan(gap) | (gap >= TRIP_GAP_S) | ((gap >= 150) & (np.nan_to_num(dist, nan=0) < SAME_STOP_M) & (gap > 0))
        g["is_trip_start"] = start
        g["trip_no"] = np.cumsum(start) - 1
        g["pos_in_trip"] = g.groupby("trip_no").cumcount()
        g["trip_len"] = g.groupby("trip_no")["stop_id"].transform("size")
        g["is_trip_end"] = g["pos_in_trip"] == g["trip_len"] - 1
        # ключ места остановки (для поиска той же остановки в прошлых рейсах)
        g["loc_key"] = (np.round(g["lon"].to_numpy(), 4) * 1e4).astype("int64") * 10_000_000 + (
            np.round(g["lat"].to_numpy(), 4) * 1e4
        ).astype("int64")
        out[int(tr)] = g
    return out
