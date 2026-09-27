"""Сквозной прогон всей системы: реплей NDTP -> ndtp.server -> backend -> ML-сервис.

Поднимает ML-сервис (:8001), backend (:8000) и NDTP-сервер (:9201), проигрывает
реальный test-день (04:00–07:30 UTC) с ускорением и проверяет API backend: ТС на карте,
инциденты, карточку, what-if и живые метрики качества (факт — только для оценки).

Запуск: ``python tests/e2e_backend.py``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACK, ML = "http://127.0.0.1:8000", "http://127.0.0.1:8001"


def http(method: str, url: str, body=None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def wait(url: str, n: int = 80):
    for _ in range(n):
        try:
            return http("GET", url)
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError(f"не поднялся {url}")


def main():
    env = {**os.environ, "ML_URL": ML, "SCHEDULE_SPLIT": "test", "PREDICT_EVERY_S": "1.5"}
    procs = []
    try:
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "services.ml_service.app:app", "--port", "8001", "--log-level", "warning"], cwd=ROOT, env=env))
        wait(ML + "/health")
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--log-level", "warning"], cwd=ROOT, env=env))
        wait(BACK + "/api/v1/health")
        procs.append(subprocess.Popen([sys.executable, "-m", "ndtp.server", "--port", "9201", "--sink", BACK + "/api/v1/telemetry"], cwd=ROOT, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2)
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "ndtp.replay", "--split", "test", "--port", "9201", "--speed", "450",
                        "--start", "2026-01-06 04:00:00", "--end", "2026-01-06 07:30:00"], cwd=ROOT, check=True)
        time.sleep(6)
        print(f"реплей 3.5 ч потока за {time.time() - t0:.0f} c")
        h = http("GET", BACK + "/api/v1/health")
        print("health:", {k: h[k] for k in ("status", "received", "cycles", "vehicles", "open_alerts", "buffered")}, "| ML:", h["ml"])
        veh = http("GET", BACK + "/api/v1/vehicles")
        sched = [v for v in veh if v["scheduled"]]
        print(f"ТС на карте: {len(veh)} (по расписанию {len(sched)}); риски:", {r: sum(v['risk'] == r for v in sched) for r in ('green', 'yellow', 'red', 'unknown')})
        alerts = http("GET", BACK + "/api/v1/alerts?status=all")
        print(f"инцидентов за прогон: {len(alerts)} (активных {sum(a['status'] != 'resolved' for a in alerts)})")
        if alerts:
            card = http("GET", BACK + f"/api/v1/alerts/{alerts[0]['id']}")
            print("карточка:", card["severity"], "|", card["prediction"]["explanation"][:160])
            print("   цель:", card["target"]["address"], "| точек участка:", len(card["segment"]), "| паттерны:", card["prediction"]["patterns"])
            tr = card["tr_id"]
            wi = http("POST", BACK + "/api/v1/whatif", {"tr_id": tr, "reserve_vehicle": True})
            print("what-if «резервное ТС по графику»: изменение прогноза", wi["delta_s"], "с")
        m = http("GET", BACK + "/api/v1/metrics")
        print("метрики:", json.dumps(m, ensure_ascii=False))
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
