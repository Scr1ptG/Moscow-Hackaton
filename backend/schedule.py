"""Расписание для backend: остановки, маршруты (линии для карты), участок до цели.

Backend читает те же CSV, что и ML-сервис, но использует их только для отображения
(карта, карточка инцидента) и для живой оценки качества. Факт прибытия
(``time_fact_begin``), если он есть в файле, **никогда не передаётся в ML** — он нужен
лишь для метрик на реплее уже прошедшего дня.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _unix(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="mixed") - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


@dataclass
class Schedule:
    """План дня по ТС + справочники остановок."""

    stops: pd.DataFrame  # stop_id, tr_id, plan, lon, lat, address, fact (NaN если нет)
    unit_to_tr: dict[int, int]

    @classmethod
    def load(cls, data_dir: str, split: str) -> "Schedule":
        d = Path(data_dir) / split
        path = d / ("schedule_plan.csv" if (d / "schedule_plan.csv").exists() else "schedule.csv")
        df = pd.read_csv(path)
        xy = df["geom"].str.extract(r"POINT \(([-\d.eE]+) ([-\d.eE]+)\)").astype(float)
        stops = pd.DataFrame(
            {
                "stop_id": df["tt_action_item_id"].astype("int64"),
                "tr_id": df["tr_id"].astype("int64"),
                "plan": _unix(df["time_begin"]),
                "lon": xy[0],
                "lat": xy[1],
                "address": df.get("building_address", pd.Series([None] * len(df))).fillna(""),
                "fact": _unix(df["time_fact_begin"]) if "time_fact_begin" in df else np.nan,
            }
        ).sort_values(["tr_id", "plan", "stop_id"], kind="stable").reset_index(drop=True)
        unit_to_tr = {}
        traffic = d / "traffic.csv"
        if traffic.exists():
            u = pd.read_csv(traffic, usecols=["unit_id", "tr_id"]).drop_duplicates()
            unit_to_tr = dict(zip(u["unit_id"].astype(int), u["tr_id"].astype(int)))
        return cls(stops=stops, unit_to_tr=unit_to_tr)

    @property
    def has_facts(self) -> bool:
        return bool(self.stops["fact"].notna().any())

    def vehicle_ids(self) -> list[int]:
        return sorted(self.stops["tr_id"].unique().tolist())

    def stop_info(self, stop_id: int) -> dict | None:
        r = self.stops[self.stops["stop_id"] == stop_id]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"stop_id": int(r.stop_id), "address": r.address, "lon": float(r.lon), "lat": float(r.lat), "plan": float(r.plan)}

    def fact_of(self, stop_id: int) -> float | None:
        r = self.stops.loc[self.stops["stop_id"] == stop_id, "fact"]
        return None if r.empty or not np.isfinite(r.iat[0]) else float(r.iat[0])

    def route_line(self, tr_id: int) -> list[list[float]]:
        """Линия маршрута для карты: уникальные остановки ТС в порядке первого проезда."""
        g = self.stops[self.stops["tr_id"] == tr_id]
        key = (g["lon"].round(4).astype(str) + "," + g["lat"].round(4).astype(str))
        first = g[~key.duplicated()]
        return first[["lon", "lat"]].round(6).to_numpy().tolist()

    def segment(self, tr_id: int, now: float, target_stop_id: int) -> list[list[float]]:
        """Участок маршрута от последней плановой остановки (<= now) до целевой — для карточки."""
        g = self.stops[self.stops["tr_id"] == tr_id].reset_index(drop=True)
        j = g.index[g["stop_id"] == target_stop_id]
        if not len(j):
            return []
        j = int(j[0])
        i = max(0, int(np.searchsorted(g["plan"].to_numpy(), now, side="right")) - 1)
        i = min(i, j)  # цель уже в прошлом (запоздалый прогноз) — показываем хотя бы её
        return g.loc[i:j, ["lon", "lat"]].round(6).to_numpy().tolist()
