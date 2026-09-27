"""Генерация разметки из сырых данных АСДУ — основа дообучения на новых днях.

Новые данные не придут с готовыми ``labels_*.csv``: придут телеметрия и
расписание с фактом (``time_fact_begin``). Правила разметки восстановлены из
выданных данных и проверены (см. :func:`verify`):

* прогнозные моменты ``T`` — сетка 5 мин;
* цель — первая остановка ТС с плановым временем в ``(T+10 мин, T+15 мин]``;
* ``cur_dev_s`` — отклонение (факт − план) на последней остановке с планом ``<= T``
  (0, если такой нет);
* ``y`` = факт − план на целевой остановке.

Запуск проверки: ``python -m ml.labels``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_points, load_schedule_facts, load_schedule_plan

GRID_S = 300
LEAD_MIN_S, LEAD_MAX_S = 600, 900


def make_labels(plan: pd.DataFrame, facts: pd.DataFrame, grid_s: int = GRID_S) -> pd.DataFrame:
    """Прогнозные точки и таргеты по плану и факту (формат как у :func:`ml.data.load_points`).

    :param plan: плановое расписание (``stop_id, tr_id, plan, ...``).
    :param facts: факт (``stop_id, fact``); строки без факта игнорируются как цели.
    """
    s = plan.merge(facts[["stop_id", "fact"]], on="stop_id", how="left")
    s["dev"] = s["fact"] - s["plan"]
    rows = []
    for tr, g in s.groupby("tr_id"):
        g = g.sort_values(["plan", "stop_id"], kind="stable")
        p, dev, ids = g["plan"].to_numpy(), g["dev"].to_numpy(), g["stop_id"].to_numpy()
        t0 = np.floor(p.min() / grid_s) * grid_s - LEAD_MAX_S
        for T in np.arange(t0, p.max(), grid_s):
            m = np.flatnonzero((p > T + LEAD_MIN_S) & (p <= T + LEAD_MAX_S))
            if not len(m) or np.isnan(dev[m[0]]):
                continue
            j = m[0]
            i_last = int(np.searchsorted(p, T, side="right")) - 1
            cur = dev[i_last] if i_last >= 0 and not np.isnan(dev[i_last]) else 0.0
            rows.append(
                dict(sample_id=f"{tr}_{int(T)}", tr_id=int(tr), T=float(T), target_stop_id=int(ids[j]),
                     target_plan=float(p[j]), cur_dev_s=float(cur), y=float(dev[j]))
            )
    out = pd.DataFrame(rows)
    out["is_synthetic"] = out["tr_id"] >= 9_000_000
    return out


def verify(split: str = "test") -> dict:
    """Сверка сгенерированной разметки с выданной организаторами."""
    gen = make_labels(load_schedule_plan(split), load_schedule_facts(split)).set_index("sample_id")
    ref = load_points(split).set_index("sample_id")
    common = ref.index.intersection(gen.index)
    g, r = gen.loc[common], ref.loc[common]
    return {
        "reference_points": int(len(ref)),
        "generated_points": int(len(gen)),
        "reference_covered": float(len(common) / len(ref)),
        "target_stop_match": float(np.mean(g["target_stop_id"] == r["target_stop_id"])),
        "y_match": float(np.mean(np.isclose(g["y"], r["y"]))),
        "cur_dev_match": float(np.mean(np.isclose(g["cur_dev_s"], r["cur_dev_s"]))),
    }


if __name__ == "__main__":
    for sp in ("test", "train"):
        print(sp, verify(sp))
