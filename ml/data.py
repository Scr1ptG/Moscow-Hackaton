"""Загрузка сырых данных хакатона.

Все времена переводятся в Unix-секунды (``float64``, наивное время раздачи
трактуется как UTC — так же, как в ``sample_id``). Работа с числами вместо
``datetime`` упрощает и ускоряет расчёт признаков.

Принцип анти-утечки: функции, возвращающие расписание для признаков
(:func:`load_schedule_plan`), **никогда** не отдают фактические времена
(``time_fact_begin``) и флаг ``manual_fill`` — они описывают будущее относительно
момента прогноза. Факт доступен только через :func:`load_schedule_facts` и
используется исключительно для калибровки/оценки, не для признаков.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("MT_DATA_DIR", r"C:\Users\ibrag\Downloads\dataset"))

#: ID синтетических ТС начинаются с этого значения (копии реальных со сдвигом).
SYNTHETIC_TR_ID_MIN = 9_000_000

SPLITS = ("train", "test", "validate")


def to_unix(s: pd.Series) -> np.ndarray:
    """Переводит строку/серию дат в Unix-секунды (float64)."""
    dt = pd.to_datetime(s, format="mixed")
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype="float64")


def parse_point_wkt(geom: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Разбирает WKT ``POINT (lon lat)`` в два массива."""
    xy = geom.str.extract(r"POINT \(([-\d.eE]+) ([-\d.eE]+)\)").astype("float64")
    return xy[0].to_numpy(), xy[1].to_numpy()


def load_traffic(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Телеметрия (раскодированные ячейки NDTP ``G6CellNav00``).

    Возвращает колонки: ``tr_id, t, valid, lon, lat, speed, heading, is_hist``.
    Невалидные координаты обнуляются в NaN (у части невалидных строк координаты
    заполнены мусором/последним известным значением).
    """
    path = data_dir / split / "traffic.csv"
    df = pd.read_csv(
        path,
        usecols=["tr_id", "event_time", "location_valid", "lon", "lat", "speed", "heading", "is_hist_data"],
        dtype={"tr_id": "int64", "lon": "float64", "lat": "float64", "speed": "float64", "heading": "float64"},
    )
    out = pd.DataFrame(
        {
            "tr_id": df["tr_id"].to_numpy(),
            "t": to_unix(df["event_time"]),
            "valid": df["location_valid"].astype(bool).to_numpy(),
            "lon": df["lon"].to_numpy(),
            "lat": df["lat"].to_numpy(),
            "speed": df["speed"].to_numpy(),
            "heading": df["heading"].to_numpy(),
            "is_hist": df["is_hist_data"].astype(bool).to_numpy(),
        }
    )
    bad = (~out["valid"]) | (out["lon"].abs() < 1) | (out["lat"].abs() < 1)
    out.loc[bad, ["lon", "lat", "heading"]] = np.nan
    out.loc[bad | (out["speed"] > 150), "speed"] = np.nan
    out["valid"] = ~bad
    out = out.sort_values(["tr_id", "t"], kind="stable").drop_duplicates(["tr_id", "t"], keep="last")
    return out.reset_index(drop=True)


def _schedule_path(split: str, data_dir: Path) -> Path:
    name = "schedule_plan.csv" if split == "validate" else "schedule.csv"
    return data_dir / split / name


def load_schedule_plan(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Плановое расписание без факта: ``stop_id, tr_id, plan, lon, lat``.

    Это единственный вид расписания, который допускается в признаки.
    """
    df = pd.read_csv(_schedule_path(split, data_dir), usecols=["tt_action_item_id", "tr_id", "time_begin", "geom"])
    lon, lat = parse_point_wkt(df["geom"])
    out = pd.DataFrame(
        {
            "stop_id": df["tt_action_item_id"].astype("int64").to_numpy(),
            "tr_id": df["tr_id"].astype("int64").to_numpy(),
            "plan": to_unix(df["time_begin"]),
            "lon": lon,
            "lat": lat,
        }
    )
    return out


def load_schedule_facts(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Факт прибытия (только train/test). **Не использовать в признаках.**

    Нужен для калибровки детектора прибытий и анализа ошибок.
    """
    if split == "validate":
        raise ValueError("В validate факта нет (и использовать его нельзя).")
    df = pd.read_csv(_schedule_path(split, data_dir), usecols=["tt_action_item_id", "time_fact_begin", "manual_fill"])
    return pd.DataFrame(
        {
            "stop_id": df["tt_action_item_id"].astype("int64").to_numpy(),
            "fact": to_unix(df["time_fact_begin"]),
            "manual_fill": df["manual_fill"].astype(bool).to_numpy(),
        }
    )


def load_points(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Прогнозные точки: разметка (train/test) или ``validate/points.csv``.

    Колонки: ``sample_id, tr_id, T, target_stop_id, target_plan, cur_dev_s``
    и, если есть, ``y`` (= ``target_delay_s``).
    """
    path = data_dir / ("validate/points.csv" if split == "validate" else f"labels/labels_{split}.csv")
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "tr_id": df["tr_id"].astype("int64").to_numpy(),
            "T": to_unix(df["T"]),
            "target_stop_id": df["target_stop_id"].astype("int64").to_numpy(),
            "target_plan": to_unix(df["target_time_begin"]),
            "cur_dev_s": df["cur_dev_s"].astype("float64").to_numpy(),
        }
    )
    if "target_delay_s" in df:
        out["y"] = df["target_delay_s"].astype("float64").to_numpy()
    out["split"] = split
    out["is_synthetic"] = out["tr_id"] >= SYNTHETIC_TR_ID_MIN
    return out


@dataclass
class SplitData:
    """Все входы одного сплита, достаточные для построения признаков."""

    name: str
    traffic: pd.DataFrame
    schedule: pd.DataFrame
    points: pd.DataFrame


def load_split(split: str, data_dir: Path = DATA_DIR) -> SplitData:
    """Загружает телеметрию, план и прогнозные точки сплита."""
    return SplitData(
        name=split,
        traffic=load_traffic(split, data_dir),
        schedule=load_schedule_plan(split, data_dir),
        points=load_points(split, data_dir),
    )


def base_vehicle_map(train_schedule: pd.DataFrame) -> dict[int, int]:
    """Сопоставляет синтетическому ТС его реальный «оригинал».

    Синтетика — копия реального ТС: те же остановки (``lon/lat``) в том же
    порядке, но со сдвигом плана по времени. Для честной групповой
    кросс-валидации копии должны лежать в одном фолде с оригиналом.
    """
    sig = {}
    for tr, g in train_schedule.sort_values(["tr_id", "plan"]).groupby("tr_id"):
        key = (len(g), round(float(np.nansum(g["lon"])), 5), round(float(np.nansum(g["lat"])), 5))
        sig[int(tr)] = key
    real = {v: k for k, v in sig.items() if k < SYNTHETIC_TR_ID_MIN}
    return {tr: (tr if tr < SYNTHETIC_TR_ID_MIN else real.get(key, tr)) for tr, key in sig.items()}
