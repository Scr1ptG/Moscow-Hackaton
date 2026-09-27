"""Обучение и оценка моделей (LightGBM / CatBoost), честная кросс-валидация.

Схема валидации — **leave-one-vehicle-out**: реальное ТС и его синтетические
копии всегда в одном фолде. Синтетика в train — это сдвинутые по времени копии
реальных ТС на весь день (в т.ч. на периоды test/validate), поэтому случайное
разбиение или «train -> test» дают оптимистичную оценку. Метрика считается
только по реальным ТС (validate полностью реальный).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Колонки, которые не являются признаками модели.
META_COLS = ["sample_id", "tr_id", "T", "target_stop_id", "target_plan", "y", "split", "is_synthetic", "base_tr", "fold"]
#: Признаки, исключённые сознательно: идентификаторы и «абсолютное время»
#: позволяют модели запоминать соседние по времени размеченные блоки того же ТС
#: (и его синтетических копий), а не обобщать.
DROP_FEATURES = ["trip_no"]


def feature_columns(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    """Список признаков модели (всё, кроме служебных и исключённых колонок)."""
    drop = set(DROP_FEATURES + (drop or []))
    return [c for c in df.columns if c not in META_COLS and c not in drop]


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def score(y, p, mae_target_ratio: float = 0.759) -> float:
    """Оценка скора платформы.

    ``MAE_TARGET`` неизвестен; из условия «бейзлайн cur_dev_s даёт ~0.40»
    на test он оценивается как ~0.76 * mae_zero.
    """
    mz = mae(y, 0)
    m = mae(y, p)
    mt = mae_target_ratio * mz
    return float(np.clip((mz - m) / (mz - mt), 0, 1))


@dataclass
class CVResult:
    """Out-of-fold прогнозы и метрики."""

    oof: np.ndarray
    metrics: dict = field(default_factory=dict)


def lgb_params(**kw) -> dict:
    p = dict(
        objective="l1",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=40,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=0,
        num_threads=8,
    )
    p.update(kw)
    return p


def fit_lgb(X, y, w=None, params=None, n_rounds=600):
    import lightgbm as lgb

    ds = lgb.Dataset(X, y, weight=w, free_raw_data=False)
    return lgb.train(params or lgb_params(), ds, num_boost_round=n_rounds)


def fit_cat(X, y, w=None, params=None):
    from catboost import CatBoostRegressor

    p = dict(loss_function="MAE", iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=0, verbose=0, thread_count=8,
             allow_writing_files=False)
    p.update(params or {})
    m = CatBoostRegressor(**p)
    m.fit(X, y, sample_weight=w)
    return m


def run_cv(
    df: pd.DataFrame,
    feats: list[str],
    fit_fn,
    syn_weight: float = 1.0,
    target: str = "y",
    base: str | None = None,
) -> CVResult:
    """Leave-one-vehicle-out CV. ``base`` — колонка базового прогноза (для остатков)."""
    oof = np.full(len(df), np.nan)
    w_all = np.where(df["is_synthetic"], syn_weight, 1.0)
    for g in sorted(df["base_tr"].unique()):
        te = (df["base_tr"] == g).to_numpy() & ~df["is_synthetic"].to_numpy()
        tr = (df["base_tr"] != g).to_numpy() & (w_all > 0)
        if te.sum() == 0:
            continue
        yt = df[target].to_numpy()
        off_tr = df[base].to_numpy()[tr] if base else 0.0
        off_te = df[base].to_numpy()[te] if base else 0.0
        m = fit_fn(df.loc[tr, feats], yt[tr] - off_tr, w_all[tr])
        oof[te] = m.predict(df.loc[te, feats]) + off_te
    real = ~df["is_synthetic"].to_numpy()
    y = df["y"].to_numpy()
    res = CVResult(oof=oof)
    res.metrics = {
        "mae_real": mae(y[real], oof[real]),
        "mae_test": mae(y[(df["split"] == "test").to_numpy()], oof[(df["split"] == "test").to_numpy()]),
        "score_real": score(y[real], oof[real]),
        "score_test": score(y[(df["split"] == "test").to_numpy()], oof[(df["split"] == "test").to_numpy()]),
    }
    return res
