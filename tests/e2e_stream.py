"""Сквозной смоук-тест онлайн-контура: NDTP-реплей -> ndtp.server -> ML-сервис -> прогнозы.

Поднимает ML-сервис и NDTP-сервер подпроцессами, проигрывает реальный test-день
(с начала суток до ``END``) NDTP-пакетами и сверяет онлайн-прогнозы для
размеченных точек с офлайн-прогнозами тех же моделей.

Запуск: ``python tests/e2e_stream.py``
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ML = "http://127.0.0.1:8001"
END = "2026-01-06 09:00:00"


def http(method: str, url: str, body=None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main():
    from ml.cache import get_features
    from ml.data import to_unix
    from ml.predictor import load_predictor

    procs = []
    try:
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "services.ml_service.app:app", "--port", "8001", "--log-level", "warning"], cwd=ROOT))
        for _ in range(60):
            try:
                http("GET", ML + "/health")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        print("schedule:", http("POST", ML + "/v1/schedule", {"split": "test"}))
        procs.append(subprocess.Popen([sys.executable, "-m", "ndtp.server", "--port", "9201", "--sink", ML + "/v1/telemetry"], cwd=ROOT))
        time.sleep(2)
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "ndtp.replay", "--split", "test", "--port", "9201", "--speed", "2000", "--end", END], cwd=ROOT, check=True)
        time.sleep(4)  # дождаться сброса батчей
        print(f"реплей занял {time.time() - t0:.1f} c; health:", http("GET", ML + "/health"))
        now = float(to_unix(pd.Series([END]))[0])
        res = http("GET", ML + f"/v1/predictions?now={now}")
        print(f"/v1/predictions: {len(res['items'])} ТС, latency {res['latency_ms']} мс")
        for it in res["items"][:5]:
            print("  ", {k: it.get(k) for k in ("tr_id", "cur_dev_s", "delay_pred", "p_late", "risk", "cause")})
            print("     ", it.get("explanation"))
        # сверка онлайн vs офлайн на размеченных точках test с T <= END
        from ml.cache import get_sequences

        lab = get_features("test")
        seq = get_sequences("test")
        pts = lab[lab["T"] <= now].tail(15)
        pr = load_predictor()
        off = pr.predict_frame(pts, seq[pts.index.to_numpy()] if pr.nn else None, explain=False)["delay_pred"].to_numpy()
        on = []
        for p in pts.itertuples():
            r = http("POST", ML + "/v1/predict", dict(tr_id=int(p.tr_id), T=float(p.T), target_stop_id=int(p.target_stop_id), target_plan=float(p.target_plan), cur_dev_s=float(p.cur_dev_s)))
            on.append(r["delay_pred"])
        on = np.array(on)
        print("онлайн vs офлайн (15 точек): max |diff| =", round(float(np.max(np.abs(on - off))), 2), "с")
        print("факт:", pts["y"].round().tolist())
        print("онлайн:", np.round(on).tolist())
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
