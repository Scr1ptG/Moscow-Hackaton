"""Жизненный цикл инцидентов: открыт -> (подтверждён диспетчером) -> закрыт.

Один активный инцидент на ТС: он обновляется, пока риск держится, и закрывается,
когда прогноз N циклов подряд «зелёный» или цель уже в прошлом без свежего прогноза.
"""
from __future__ import annotations

from .config import Settings
from .schedule import Schedule
from .state import Alert, Store

RISK_ORDER = {"green": 0, "yellow": 1, "red": 2}


def alert_summary(a: Alert) -> dict:
    p = a.prediction
    return {
        "id": a.id, "tr_id": a.tr_id, "status": a.status, "severity": a.severity,
        "opened_at": a.opened_at, "updated_at": a.updated_at, "resolved_at": a.resolved_at,
        "delay_pred": p.get("delay_pred"), "p_late": p.get("p_late"), "cause": p.get("cause"),
        "target_address": (a.target or {}).get("address") or (f"остановка №{a.target['stop_id']}" if a.target else None),
        "peak_delay_s": a.peak_delay_s,
    }


def alert_card(a: Alert) -> dict:
    """Карточка инцидента для диспетчера: прогноз, интервал, причина, объяснение, участок."""
    p = a.prediction
    keep = ("delay_pred", "q10", "q90", "p_late", "expected_abs_error", "risk", "cause", "cause_code", "explanation",
            "top_features", "patterns", "recommendations", "base_value", "rule_contrib", "target_stop_id", "target_plan", "cur_dev_s", "T")
    return {
        **alert_summary(a),
        "prediction": {k: p.get(k) for k in keep},
        "contributions": {k.replace("contrib_", ""): v for k, v in p.items() if k.startswith("contrib_")},
        "target": a.target,
        "segment": a.segment,
        "history": [{"t": t, "delay_pred": d, "risk": r} for t, d, r in a.history[-60:]],
    }


def update_alerts(store: Store, schedule: Schedule, items: list[dict], now: float, cfg: Settings) -> list[dict]:
    """Обновляет инциденты по свежим прогнозам; возвращает события для WebSocket."""
    events = []
    min_rank = RISK_ORDER.get(cfg.alert_min_risk, 1)
    seen = set()
    for it in items:
        tr = int(it["tr_id"])
        seen.add(tr)
        risk = it.get("risk", "green")
        aid = store.open_alert_by_tr.get(tr)
        if RISK_ORDER.get(risk, 0) >= min_rank:
            target = schedule.stop_info(int(it["target_stop_id"]))
            seg = schedule.segment(tr, now, int(it["target_stop_id"]))
            if aid is None:
                a = Alert(tr_id=tr, severity=risk, opened_at=now, prediction=it, target=target, segment=seg, updated_at=now)
                store.alerts[a.id] = a
                store.open_alert_by_tr[tr] = a.id
                events.append({"type": "alert_opened", "alert": alert_summary(a)})
            else:
                a = store.alerts[aid]
                changed = a.severity != risk
                a.severity, a.prediction, a.target, a.segment = risk, it, target, seg
                a.updated_at, a.green_cycles = now, 0
                if changed:
                    events.append({"type": "alert_updated", "alert": alert_summary(a)})
            a.peak_delay_s = max(a.peak_delay_s, float(it.get("delay_pred") or 0))
            a.history.append((now, it.get("delay_pred"), risk))
        elif aid is not None:
            a = store.alerts[aid]
            a.green_cycles += 1
            a.history.append((now, it.get("delay_pred"), risk))
            if a.green_cycles >= cfg.resolve_after_cycles:
                events.append(_resolve(store, a, now, "risk_cleared"))
    # цель инцидента в прошлом, а свежего прогноза по ТС нет (отстой на конечной, потеря связи)
    for tr, aid in list(store.open_alert_by_tr.items()):
        a = store.alerts[aid]
        plan = (a.target or {}).get("plan")
        if tr not in seen and plan is not None and now > plan + 600:
            events.append(_resolve(store, a, now, "target_passed"))
    return events


def _resolve(store: Store, a: Alert, now: float, reason: str) -> dict:
    a.status, a.resolved_at = "resolved", now
    store.open_alert_by_tr.pop(a.tr_id, None)
    return {"type": "alert_resolved", "reason": reason, "alert": alert_summary(a)}
