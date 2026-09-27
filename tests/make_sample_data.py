"""Синтетический мини-датасет в формате раздачи (для смоук-теста Docker в CI без реальных данных).

Два ТС ходят туда-обратно по линии из 10 остановок (два рейса с отстоем на конечной);
телеметрия каждые 15 с, с растущей задержкой — достаточно, чтобы вся цепочка
NDTP -> backend -> ML дала прогнозы и инциденты. Только стандартная библиотека.

Запуск: ``python tests/make_sample_data.py sample_data`` -> ``sample_data/test/{schedule,traffic}.csv``
"""
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

T0 = datetime(2026, 1, 6, 6, 0, 0)
STOPS = [(37.60 + 0.005 * i, 55.750 + 0.001 * i) for i in range(10)]  # ~350 м между остановками


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def main(out_dir: str = "sample_data"):
    d = Path(out_dir) / "test"
    d.mkdir(parents=True, exist_ok=True)
    sched, traffic = [], []
    item = 10_000
    for v, (tr_id, unit_id, shift_min, drift) in enumerate([(1001, 5001, 0, 12.0), (1002, 5002, 5, 20.0)]):
        visits = []  # (плановое время, факт, координаты)
        t = T0 + timedelta(minutes=shift_min)
        for trip in range(2):
            order = list(range(10)) if trip % 2 == 0 else list(range(9, -1, -1))
            for k, i in enumerate(order):
                plan = t + timedelta(seconds=90 * k)
                fact = plan + timedelta(seconds=30 + drift * k)  # задержка растёт вдоль рейса
                visits.append((plan, fact, STOPS[i]))
            t = t + timedelta(seconds=90 * 9 + 6 * 60)  # отстой 6 мин на конечной
        for plan, fact, (lon, lat) in visits:
            item += 1
            sched.append([item, _fmt(plan), _fmt(fact), "2026-01-06", "False", tr_id, f"POINT ({lon} {lat})", f"Тестовая ул., д.{item % 50}"])
        # трек: линейно между фактическими прибытиями, 20 с стоянка на остановке
        start, end = visits[0][1] - timedelta(minutes=5), visits[-1][1] + timedelta(minutes=5)
        ts, seq = start, 0
        while ts <= end:
            prev = max((x for x in visits if x[1] <= ts), key=lambda x: x[1], default=visits[0])
            nxt = min((x for x in visits if x[1] > ts), key=lambda x: x[1], default=visits[-1])
            if nxt[1] == prev[1] or (ts - prev[1]).total_seconds() < 20:
                lon, lat, speed = prev[2][0], prev[2][1], 0.0
            else:
                u = min(1.0, (ts - prev[1] - timedelta(seconds=20)) / max(nxt[1] - prev[1] - timedelta(seconds=20), timedelta(seconds=1)))
                lon = prev[2][0] + u * (nxt[2][0] - prev[2][0])
                lat = prev[2][1] + u * (nxt[2][1] - prev[2][1])
                speed = 25.0
            seq += 1
            traffic.append([f"{v}{seq}", tr_id, unit_id, _fmt(ts), 0, "True", _fmt(ts), round(lon, 7), round(lat, 7), 150, speed,
                            round(math.degrees(math.atan2(1, 1)) % 360), _fmt(ts), "False"])
            ts += timedelta(seconds=15)
    with open(d / "schedule.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tt_action_item_id", "time_begin", "time_fact_begin", "order_date", "manual_fill", "tr_id", "geom", "building_address"])
        w.writerows(sched)
    with open(d / "traffic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["packet_id", "tr_id", "unit_id", "event_time", "device_event_id", "location_valid", "gps_time", "lon", "lat", "alt",
                    "speed", "heading", "receive_time", "is_hist_data"])
        w.writerows(traffic)
    print(f"{d}: {len(sched)} остановок, {len(traffic)} точек телеметрии")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
