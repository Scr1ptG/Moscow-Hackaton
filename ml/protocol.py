"""Честный протокол оценки модели (защита от утечки через синтетические копии и от переобучения).

Две независимые проверки, изменение принимается, только если помогает в **обеих**
и сильнее шума (разброс по сидам):

* **P1 — leave-one-vehicle-out** по реальным ТС. В обучении нет ни самого ТС, ни
  его синтетических копий. Вариант ``synthetic=False`` — обучение только на реальных ТС.
* **P2 — train → test без синтетики**: обучение на реальных точках train, проверка на
  test (как validate, но без копий периода проверки в обучении).

Синтетика в train — копии реальных ТС на весь день; в финальном обучении под validate она
дала бы модели «подсмотреть» ответы. Поэтому честная финальная модель обучается без неё.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .compact import postprocess
from .models import mae

FitFn = Callable[[pd.DataFrame, np.ndarray, int], object]  # (X, y, seed) -> модель с .predict


def lovo(lab: pd.DataFrame, feats: list[str], fit: FitFn, seeds=(0, 1, 2), synthetic: bool = True,
         target: Callable[[pd.DataFrame], np.ndarray] | None = None,
         inverse: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None) -> dict:
    """P1: out-of-fold прогноз по реальным ТС. ``target``/``inverse`` — для остаточных таргетов."""
    real = ~lab["is_synthetic"].to_numpy()
    pool = lab if synthetic else lab[real]
    per_seed = []
    for s in seeds:
        oof = np.full(len(lab), np.nan)
        for g in sorted(lab["base_tr"].unique()):
            te = (lab["base_tr"] == g).to_numpy() & real
            tr = pool[pool["base_tr"] != g]
            yt = target(tr) if target else tr["y"].to_numpy()
            p = fit(tr[feats], yt, s).predict(lab.loc[te, feats])
            oof[te] = inverse(lab[te], p) if inverse else p
        per_seed.append(postprocess(oof, lab["tgt_is_trip_start"].to_numpy()))
    y = lab["y"].to_numpy()
    t = (lab["split"] == "test").to_numpy()
    m_real = [mae(y[real], o[real]) for o in per_seed]  # качество ОДНОЙ модели (как в проде)
    m_test = [mae(y[t], o[t]) for o in per_seed]
    return {
        "mae_real": round(float(np.mean(m_real)), 2),
        "mae_test": round(float(np.mean(m_test)), 2),
        "seed_spread": round(float(np.std(m_real)), 2),
        "oof": np.mean(per_seed, axis=0),
    }


def train_to_test(lab: pd.DataFrame, feats: list[str], fit: FitFn, seeds=(0, 1, 2),
                  target: Callable[[pd.DataFrame], np.ndarray] | None = None,
                  inverse: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None) -> float:
    """P2: обучение на реальных точках train, MAE на test."""
    tr = lab[(lab["split"] == "train") & ~lab["is_synthetic"]]
    te = lab[lab["split"] == "test"]
    yt = target(tr) if target else tr["y"].to_numpy()
    out = []
    for s in seeds:  # среднее качество одной модели по сидам
        p = fit(tr[feats], yt, s).predict(te[feats])
        p = inverse(te, p) if inverse else p
        out.append(mae(te["y"], postprocess(p, te["tgt_is_trip_start"].to_numpy())))
    return round(float(np.mean(out)), 2)


def evaluate(lab: pd.DataFrame, feats: list[str], fit: FitFn, **kw) -> dict:
    """Обе проверки одной строкой: P1 (с синтетикой других ТС и без неё) и P2."""
    a = lovo(lab, feats, fit, synthetic=True, **kw)
    b = lovo(lab, feats, fit, synthetic=False, **kw)
    return {
        "P1_lovo": a["mae_real"], "P1_lovo_test": a["mae_test"], "P1_spread": a["seed_spread"],
        "P1_lovo_real_only": b["mae_real"],
        "P2_train_to_test": train_to_test(lab, feats, fit, **kw),
    }


# ------------------------------------------------------------------ «чистая» синтетика


def synthetic_shifts() -> dict[int, float]:
    """Сдвиг времени каждой синтетической копии относительно её реального ТС, с.

    Копия — то же расписание, сдвинутое на константу (проверено: std сдвига = 0).
    """
    from .data import base_vehicle_map, load_schedule_plan

    plan = load_schedule_plan("train")
    bm = base_vehicle_map(plan)
    out = {}
    for syn, real in bm.items():
        if syn == real:
            continue
        a = plan[plan["tr_id"] == real].sort_values(["plan", "stop_id"])["plan"].to_numpy()
        b = plan[plan["tr_id"] == syn].sort_values(["plan", "stop_id"])["plan"].to_numpy()
        out[syn] = float(np.median(b - a))
    return out


def twin_contaminated(lab: pd.DataFrame, protected: pd.DataFrame, margin_s: float = 1800.0) -> np.ndarray:
    """Маска синтетических строк, чьё «реальное» время попадает в защищённые периоды.

    Синтетическая точка в ``T`` у копии со сдвигом Δ соответствует реальному моменту
    ``T − Δ`` её оригинала. Если он ближе ``margin_s`` к любой защищённой точке того же
    оригинала (validate или test), строка — копия ответа и из обучения исключается.
    Запас 30 мин покрывает горизонт цели (15 мин) и «историю» в признаках.
    """
    shifts = synthetic_shifts()
    prot = {g: np.sort(d["T"].to_numpy()) for g, d in protected.groupby("base_tr")}
    bad = np.zeros(len(lab), bool)
    syn = lab["is_synthetic"].to_numpy()
    for i in np.flatnonzero(syn):
        tr, base = int(lab["tr_id"].iat[i]), int(lab["base_tr"].iat[i])
        ts = prot.get(base)
        if ts is None or tr not in shifts:
            continue
        t_real = lab["T"].iat[i] - shifts[tr]
        k = np.searchsorted(ts, t_real)
        near = [abs(ts[j] - t_real) for j in (k - 1, k) if 0 <= j < len(ts)]
        bad[i] = bool(near) and min(near) <= margin_s
    return bad


def train_to_test_clean(lab: pd.DataFrame, feats: list[str], fit: FitFn, seeds=(0, 1, 2), margin_s: float = 1800.0) -> dict:
    """P2c: train (реальные + синтетика БЕЗ копий периодов test) -> test."""
    te = lab[lab["split"] == "test"]
    bad = twin_contaminated(lab, te, margin_s)
    tr = lab[(lab["split"] == "train") & ~bad]
    out = []
    for s in seeds:
        p = fit(tr[feats], tr["y"].to_numpy(), s).predict(te[feats])
        out.append(mae(te["y"], postprocess(p, te["tgt_is_trip_start"].to_numpy())))
    return {"mae": round(float(np.mean(out)), 2), "synthetic_kept": int((tr["is_synthetic"]).sum()),
            "synthetic_removed": int(bad[(lab["split"] == "train").to_numpy()].sum())}


def forward_in_time(lab: pd.DataFrame, feats: list[str], fit: FitFn, seeds=(0, 1, 2),
                    cuts_local_h=(11, 16), horizon_h: float = 5.0, margin_s: float = 1800.0) -> dict:
    """P3: «прямо по времени» — обучаемся на прошлом, проверяемся на будущем (как в эксплуатации).

    Для каждого среза: обучение на точках с ``T < cut`` (реальные + синтетика, чьё реальное
    время < cut − margin), проверка на реальных точках в ``[cut, cut + horizon)``.
    В отличие от P2, модель не видит разметку того же дня **после** проверяемого момента.
    """
    shifts = synthetic_shifts()
    hour = (lab["T"] / 3600 + 3) % 24
    real_T = lab["T"] - lab["tr_id"].map(shifts).fillna(0.0)  # «реальное» время строки
    res, pooled = {}, []
    for c in cuts_local_h:
        tr = lab[(real_T.to_numpy() < 0) | ((hour < c) & (((real_T / 3600 + 3) % 24) < c - margin_s / 3600)).to_numpy()]
        te = lab[~lab["is_synthetic"] & (hour >= c) & (hour < c + horizon_h)]
        m = []
        for s in seeds:
            p = fit(tr[feats], tr["y"].to_numpy(), s).predict(te[feats])
            e = np.abs(te["y"].to_numpy() - postprocess(p, te["tgt_is_trip_start"].to_numpy()))
            m.append(e.mean())
            if s == seeds[0]:
                pooled.append(e)
        res[f"cut_{c}h"] = round(float(np.mean(m)), 2)
    res["pooled"] = round(float(np.concatenate(pooled).mean()), 2)
    return res
