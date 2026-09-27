"""Реплеер исторической телеметрии в NDTP: ``traffic.csv`` -> TCP-поток как у бортовых терминалов.

Эмулятор из раздачи генерирует случайную «живую» навигацию; для демонстрации
на реальных треках этот реплеер шлёт строки ``traffic.csv`` теми же
NDTP-пакетами (handshake + ``NPH_SND_REALTIME`` с ячейкой ``G6CellNav00``),
с исходными метками времени и ускорением ``--speed``.

Запуск::

    python -m ndtp.replay --split test --host localhost --port 9201 --speed 60
"""
from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np
import pandas as pd

from ml.data import DATA_DIR, to_unix

from .protocol import build_handshake, build_nav00, build_realtime


async def replay(split: str, host: str, port: int, speed: float, start: str | None, units: list[int] | None, end: str | None = None):
    df = pd.read_csv(DATA_DIR / split / "traffic.csv", usecols=["unit_id", "event_time", "location_valid", "lon", "lat", "alt", "speed", "heading"])
    df["t"] = to_unix(df["event_time"])
    df = df.sort_values("t")
    if start:
        df = df[df["t"] >= to_unix(pd.Series([start]))[0]]
    if end:
        df = df[df["t"] <= to_unix(pd.Series([end]))[0]]
    if units:
        df = df[df["unit_id"].isin(units)]
    writers: dict[int, asyncio.StreamWriter] = {}
    req: dict[int, int] = {}
    t_data0, t_wall0 = df["t"].iloc[0], time.time()
    for r in df.itertuples(index=False):
        u = int(r.unit_id)
        if u not in writers:
            _, w = await asyncio.open_connection(host, port)
            w.write(build_handshake(u))
            await w.drain()
            writers[u], req[u] = w, 1
        delay = (r.t - t_data0) / speed - (time.time() - t_wall0)
        if delay > 0:
            await asyncio.sleep(delay)
        req[u] += 1
        valid = bool(r.location_valid) and np.isfinite(r.lon)
        cell = build_nav00(int(r.t), r.lon if valid else 0.0, r.lat if valid else 0.0, valid,
                           speed=0 if np.isnan(r.speed) else r.speed, course=0 if np.isnan(r.heading) else r.heading,
                           alt=0 if np.isnan(r.alt) else r.alt)
        writers[u].write(build_realtime(u, req[u], cell))
    for w in writers.values():
        await w.drain()
        w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9201)
    ap.add_argument("--speed", type=float, default=60.0, help="ускорение времени (60 = час данных за минуту)")
    ap.add_argument("--start", default=None, help="начать с момента, напр. '2026-01-06 06:00:00'")
    ap.add_argument("--end", default=None, help="закончить на моменте, напр. '2026-01-06 08:00:00'")
    ap.add_argument("--units", type=int, nargs="*", default=None)
    a = ap.parse_args()
    asyncio.run(replay(a.split, a.host, a.port, a.speed, a.start, a.units, a.end))
