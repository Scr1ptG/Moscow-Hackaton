"""Проверка анти-утечки и согласованности онлайн/офлайн признаков.

Онлайн-контекст (:class:`ml.online.OnlineState`) физически содержит только
телеметрию с ``t <= T``. Если офлайн-признаки (посчитанные на полном треке дня)
совпадают с онлайн-признаками, значит офлайн-пайплайн не заглядывает в будущее.

Запуск: ``python -m pytest tests/test_no_leak.py -q`` или ``python tests/test_no_leak.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import load_split  # noqa: E402
from ml.features import point_features, prepare_context  # noqa: E402
from ml.online import OnlineState  # noqa: E402


def _compare(split: str = "test", n: int = 120, seed: int = 0) -> tuple[int, list]:
    sp = load_split(split)
    off = prepare_context(sp)
    on = OnlineState(sp.schedule)
    tr = sp.traffic
    on.add_records(
        [dict(tr_id=int(r.tr_id), t=r.t, valid=bool(r.valid), lon=r.lon, lat=r.lat, speed=r.speed, heading=r.heading) for r in tr.itertuples()]
    )
    pts = sp.points.sample(min(n, len(sp.points)), random_state=seed)
    bad = []
    for p in pts.itertuples():
        a = point_features(off, p.tr_id, p.T, p.target_stop_id, p.target_plan, p.cur_dev_s)
        ctx = on.context(p.T, [p.tr_id])
        b = point_features(ctx, p.tr_id, p.T, p.target_stop_id, p.target_plan, p.cur_dev_s)
        for k in set(a) | set(b):
            va, vb = a.get(k, np.nan), b.get(k, np.nan)
            if not (np.isclose(va, vb, equal_nan=True, atol=1e-6)):
                bad.append((p.sample_id, k, va, vb))
    return len(pts), bad


def test_offline_equals_online():
    n, bad = _compare()
    assert not bad, f"{len(bad)} расхождений, напр. {bad[:5]}"


if __name__ == "__main__":
    n, bad = _compare()
    print(f"проверено точек: {n}, расхождений признаков: {len(bad)}")
    for b in bad[:20]:
        print(b)
