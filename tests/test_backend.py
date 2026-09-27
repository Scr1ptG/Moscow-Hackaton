"""Тесты backend с фейковым ML-сервисом: приём, пересылка, инциденты, деградация, what-if, WS.

Запуск: ``python -m pytest tests/test_backend.py -q`` (нужен датасет: расписание test-дня).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Settings  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.ml_client import CircuitBreaker, MLUnavailable  # noqa: E402

TR = 131672  # реальное ТС test-дня
NOW = 1767690000.0


def _target_for(tr: int, now: float) -> tuple[int, float]:
    """Реальная целевая остановка ТС: первая с планом в (now+10, now+15] мин."""
    from backend.schedule import Schedule

    st = Schedule.load(Settings().data_dir, "test").stops
    g = st[(st["tr_id"] == tr) & (st["plan"] > now + 600) & (st["plan"] <= now + 900)]
    return int(g["stop_id"].iat[0]), float(g["plan"].iat[0])


class FakeML:
    """Имитация ML-сервиса с тем же асинхронным интерфейсом, что у MLClient."""

    def __init__(self):
        self.breaker = CircuitBreaker()
        self.latencies_ms = [5.0]
        self.telemetry: list[dict] = []
        self.risk = "red"
        self.down = False
        self.last_whatif = None

    def _check(self):
        if self.down:
            self.breaker.failure()
            raise MLUnavailable("down")

    async def health(self):
        return {"status": "ok"}

    async def load_schedule(self, split, unit_to_tr):
        self._check()
        return {"vehicles": 13}

    async def send_telemetry(self, records):
        self._check()
        self.telemetry += records
        return {"accepted": len(records)}

    async def predictions(self, now):
        self._check()
        delay = {"red": 260.0, "yellow": 90.0, "green": 10.0}[self.risk]
        stop, plan = _target_for(TR, now)
        return {"now": now, "latency_ms": 3.0, "items": [{
            "tr_id": TR, "T": now, "target_stop_id": stop, "target_plan": plan, "cur_dev_s": 200,
            "delay_pred": delay, "q10": delay - 60, "q90": delay + 90, "p_late": 0.8 if self.risk == "red" else 0.1,
            "expected_abs_error": 60.0, "risk": self.risk, "cause": "Текущее отклонение от графика",
            "cause_code": "current_delay", "explanation": "Прогноз: опоздание …", "top_features": [], "patterns": [],
            "base_value": 41.0, "rule_contrib": 0.0, "contrib_current_delay": 150.0,
        }]}

    async def whatif(self, payload):
        self._check()
        self.last_whatif = payload
        return {"before": {"delay_pred": 260.0}, "after": {"delay_pred": 180.0}, "delta_s": -80.0}

    async def model_info(self):
        return {"kind": "compact"}

    async def reload_model(self):
        return {"reloaded": True}


@pytest.fixture()
def env():
    ml = FakeML()
    app = create_app(Settings(schedule_split="test", resolve_after_cycles=2), ml=ml, start_loops=False)
    with TestClient(app) as client:
        w = app.state.worker
        client.portal.call(w._schedule_loop)
        yield client, w, ml


def _records(n=3, t0=NOW - 60):
    return [{"tr_id": TR, "t": t0 + 15 * i, "valid": True, "lon": 37.85, "lat": 55.73, "speed": 20.0} for i in range(n)]


def test_ingest_forward_and_vehicle(env):
    client, w, ml = env
    assert client.post("/api/v1/telemetry", json=_records()).json() == {"accepted": 3}
    assert client.portal.call(w.forward_once) == 3 and len(ml.telemetry) == 3
    v = [x for x in client.get("/api/v1/vehicles").json() if x["tr_id"] == TR][0]
    assert v["scheduled"] and v["connection"] == "ok" and v["lon"] == 37.85


def test_alert_lifecycle(env):
    client, w, ml = env
    client.post("/api/v1/telemetry", json=_records())
    client.portal.call(w.forward_once)
    events = client.portal.call(w.predict_once)
    assert any(e["type"] == "alert_opened" for e in events)
    alerts = client.get("/api/v1/alerts").json()
    assert len(alerts) == 1 and alerts[0]["severity"] == "red"
    card = client.get(f"/api/v1/alerts/{alerts[0]['id']}").json()
    assert card["prediction"]["explanation"] and card["target"]["address"] and card["segment"]
    assert client.post(f"/api/v1/alerts/{alerts[0]['id']}/ack").json()["status"] == "acknowledged"
    ml.risk = "green"
    for k in range(2):  # два «зелёных» цикла -> инцидент закрыт
        client.post("/api/v1/telemetry", json=_records(1, NOW + 30 * (k + 1)))
        client.portal.call(w.forward_once)
        client.portal.call(w.predict_once)
    assert client.get("/api/v1/alerts").json() == []
    assert client.get("/api/v1/alerts?status=all").json()[0]["status"] == "resolved"


def test_ml_outage_degrades_without_crash(env):
    client, w, ml = env
    client.post("/api/v1/telemetry", json=_records())
    client.portal.call(w.forward_once)
    client.portal.call(w.predict_once)
    ml.down = True
    client.post("/api/v1/telemetry", json=_records(2, NOW + 60))
    assert client.portal.call(w.forward_once) == 0  # не доставлено -> осталось в буфере
    assert client.get("/api/v1/health").json()["buffered"] == 2
    w.ml_synced_t = NOW + 90
    assert client.portal.call(w.predict_once) == []
    h = client.get("/api/v1/health").json()
    assert h["status"] == "degraded"
    v = [x for x in client.get("/api/v1/vehicles").json() if x["tr_id"] == TR][0]
    assert v["prediction_stale"] is True and v["delay_pred"] == 260.0  # последнее известное состояние
    ml.down = False
    assert client.portal.call(w.forward_once) == 2  # после восстановления буфер дослан


def test_whatif_scenarios(env):
    client, w, ml = env
    client.post("/api/v1/telemetry", json=_records())
    client.portal.call(w.forward_once)
    client.portal.call(w.predict_once)
    r = client.post("/api/v1/whatif", json={"tr_id": TR, "extra_layover_min": 3, "reserve_vehicle": True}).json()
    assert r["delta_s"] == -80.0
    assert ml.last_whatif["shifts"] == {"layover_slack_s": 180} and ml.last_whatif["overrides"]["cur_dev_s"] == 0.0


def test_websocket_hello(env):
    client, w, ml = env
    with client.websocket_connect("/api/v1/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
