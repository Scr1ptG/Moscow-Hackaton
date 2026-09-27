r"""Запуск всей системы без Docker одной командой (ML-сервис, backend + дашборд, NDTP-сервер).

    python run_local.py                 # поднять сервисы; реплей — кнопкой в дашборде
    python run_local.py --replay 120    # сразу запустить реплей test-дня с ускорением x120

Дашборд: http://localhost:8000/  ·  Swagger backend: http://localhost:8000/docs
ML-сервис: http://localhost:8001/docs  ·  NDTP (TCP): localhost:9201
Путь к датасету — переменная окружения MT_DATA_DIR (по умолчанию C:SERSIBRAGDOWNLOADSDATASET).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def wait(url: str, n: int = 120) -> bool:
    for _ in range(n):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=float, default=0, help="ускорение реплея test-дня (0 — не запускать)")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    env = {**os.environ, "ML_URL": "http://127.0.0.1:8001", "SCHEDULE_SPLIT": a.split, "PYTHONUNBUFFERED": "1"}
    py = sys.executable
    procs = [subprocess.Popen([py, "-m", "uvicorn", "services.ml_service.app:app", "--port", "8001", "--log-level", "warning"], cwd=ROOT, env=env)]
    if not wait("http://127.0.0.1:8001/health"):
        sys.exit("ML-сервис не поднялся")
    procs.append(subprocess.Popen([py, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"], cwd=ROOT, env=env))
    if not wait("http://127.0.0.1:8000/api/v1/health"):
        sys.exit("backend не поднялся")
    procs.append(subprocess.Popen([py, "-m", "ndtp.server", "--port", "9201", "--sink", "http://127.0.0.1:8000/api/v1/telemetry"], cwd=ROOT, env=env))
    print("\n  Дашборд:  http://localhost:8000/\n  Swagger:  http://localhost:8000/docs  (backend), http://localhost:8001/docs (ML)\n  NDTP:     tcp://localhost:9201\n", flush=True)
    if a.replay:
        time.sleep(1.5)
        procs.append(subprocess.Popen([py, "-m", "ndtp.replay", "--split", a.split, "--port", "9201", "--speed", str(a.replay)], cwd=ROOT, env=env))
    try:
        while all(p.poll() is None for p in procs[:3]):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
