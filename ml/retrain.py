"""Дообучение продовой модели на новых данных (champion / challenger).

Новые данные — это просто выгрузки АСДУ за новые дни в формате раздачи:
``<dir>/traffic.csv`` + ``<dir>/schedule.csv`` (план + факт). Разметка строится
автоматически (:mod:`ml.labels`), признаки — тем же кодом, что в проде.

Режимы:

* ``full`` — переобучение с нуля на базе + новых данных (раз в сутки/неделю);
* ``warm`` — быстрое дообучение: к текущей модели достраиваются деревья на новых
  данных (LightGBM ``init_model``), минуты вместо полного цикла.

Гейт продвижения: претендент сравнивается с чемпионом на отложенном по времени
хвосте новых данных; продвигается, только если не хуже (допуск ``--tolerance``).
Все версии сохраняются в реестре (:mod:`ml.registry`), откат — одной командой.

Примеры::

    python -m ml.retrain --new-data D:/asdu/2026-01-07 --mode full --promote auto
    python -m ml.retrain --simulate            # демонстрация на данных хакатона
    python -m ml.retrain --rollback
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import registry
from .compact import FEATS, N_ROUNDS, REPORT_DIR, calibration_table, load_labeled, params, postprocess
from .data import SplitData, load_schedule_facts, load_schedule_plan, load_traffic
from .features import build_features
from .labels import make_labels
from .models import mae


def load_raw_day(path: Path) -> pd.DataFrame:
    """Признаки + таргеты для выгрузки АСДУ за новый день (``traffic.csv`` + ``schedule.csv``)."""
    path = Path(path)
    plan = load_schedule_plan(path.name, path.parent)
    facts = load_schedule_facts(path.name, path.parent)
    split = SplitData(path.name, load_traffic(path.name, path.parent), plan, make_labels(plan, facts))
    df = build_features(split)
    df["base_tr"] = df["tr_id"]
    return df


def _fit(df: pd.DataFrame, init_model=None, rounds: int = N_ROUNDS, lr: float | None = None) -> lgb.Booster:
    p = params()
    if lr:
        p["learning_rate"] = lr
    ds = lgb.Dataset(df[FEATS], df["y"].to_numpy(), free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=rounds, init_model=init_model)


def _predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return postprocess(model.predict(df[FEATS]), df["tgt_is_trip_start"].to_numpy())


def _lovo_calibration(df: pd.DataFrame) -> dict:
    """Таблица калибровки по out-of-fold остаткам (группы — ТС)."""
    oof = np.full(len(df), np.nan)
    for g in df["base_tr"].unique():
        te = (df["base_tr"] == g).to_numpy()
        if te.all():
            continue
        oof[te] = _predict(_fit(df[~te]), df[te])
    ok = ~np.isnan(oof)
    return calibration_table(oof[ok], df["y"].to_numpy()[ok])


def _write_candidate(model: lgb.Booster, calib: dict, n_train: int) -> Path:
    champion_meta = json.loads((registry.PROD_DIR / "meta.json").read_text(encoding="utf-8"))
    d = Path(tempfile.mkdtemp(prefix="challenger_"))
    model.save_model(str(d / "model.txt"))
    meta = {**champion_meta, "calibration": calib, "n_train": int(n_train)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def retrain(base: pd.DataFrame, new: pd.DataFrame, mode: str, holdout_frac: float, tolerance: float, promote: str, tag: str = "") -> dict:
    """Обучает претендента, сравнивает с чемпионом на хвосте новых данных, продвигает по гейту."""
    t_cut = new["T"].quantile(1 - holdout_frac)
    new_tr, hold = new[new["T"] < t_cut], new[(new["T"] >= t_cut) & ~new["is_synthetic"]]
    champion = lgb.Booster(model_file=str(registry.PROD_DIR / "model.txt"))
    if mode == "warm":
        challenger = _fit(new_tr, init_model=champion, rounds=100, lr=0.01)
        calib = json.loads((registry.PROD_DIR / "meta.json").read_text(encoding="utf-8"))["calibration"]
    else:
        train = pd.concat([base, new_tr], ignore_index=True)
        challenger = _fit(train)
        calib = _lovo_calibration(train)
    y = hold["y"].to_numpy()
    rep = {
        "mode": mode,
        "n_base": int(len(base)), "n_new_train": int(len(new_tr)), "n_holdout": int(len(hold)),
        "holdout_mae": {"zero": mae(y, 0), "cur_dev_s": mae(y, hold["cur_dev_s"]),
                        "champion": mae(y, _predict(champion, hold)), "challenger": mae(y, _predict(challenger, hold))},
    }
    better = rep["holdout_mae"]["challenger"] <= rep["holdout_mae"]["champion"] + tolerance
    rep["decision"] = "promote" if (promote == "force" or (promote == "auto" and better)) else ("keep champion" if not better else "not promoted (dry run)")
    if promote in ("auto", "force"):
        if mode == "full" and rep["decision"] == "promote":  # финальная модель — на всех данных, включая хвост
            challenger = _fit(pd.concat([base, new], ignore_index=True))
        vid = registry.save_version(_write_candidate(challenger, calib, len(base) + len(new)), rep, tag or mode)
        rep["version"] = vid
        if rep["decision"] == "promote":
            registry.promote(vid)
    return rep


def simulate() -> dict:
    """Демонстрация на данных хакатона: «вчерашняя» модель учится на утре, дообучается на дне,
    проверка — на вечере. Только реальные ТС (синтетические копии исказили бы оценку)."""
    lab, _ = load_labeled()
    real = lab[~lab["is_synthetic"]].copy()
    hour = (real["T"] / 3600 + 3) % 24
    morning, day = real[hour < 12], real[(hour >= 12)]
    res = {}
    champ = _fit(morning)
    tmp = Path(tempfile.mkdtemp(prefix="sim_champion_"))
    champ.save_model(str(tmp / "model.txt"))
    t_cut = day["T"].quantile(0.5)
    day_tr, hold = day[day["T"] < t_cut], day[day["T"] >= t_cut]
    y = hold["y"].to_numpy()
    full = _fit(pd.concat([morning, day_tr]))
    warm = _fit(day_tr, init_model=champ, rounds=100, lr=0.01)
    res["holdout_mae"] = {
        "cur_dev_s": mae(y, hold["cur_dev_s"]),
        "champion (только утро)": mae(y, _predict(champ, hold)),
        "warm (утро + 100 деревьев на дне)": mae(y, _predict(warm, hold)),
        "full (переобучение утро+день)": mae(y, _predict(full, hold)),
    }
    res["sizes"] = {"morning": int(len(morning)), "day_train": int(len(day_tr)), "holdout_evening": int(len(hold))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data", action="append", default=[], help="папка с traffic.csv и schedule.csv (можно несколько)")
    ap.add_argument("--mode", choices=["full", "warm"], default="full")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--tolerance", type=float, default=0.5, help="допуск ухудшения MAE, с")
    ap.add_argument("--promote", choices=["auto", "never", "force"], default="auto")
    ap.add_argument("--no-base", action="store_true", help="не добавлять исходную разметку хакатона")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(json.dumps(registry.list_versions(), ensure_ascii=False, indent=2))
        return
    if a.rollback:
        print("откат на", registry.rollback())
        return
    if a.simulate:
        rep = simulate()
        (REPORT_DIR / "retrain_simulation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return
    base = pd.DataFrame() if a.no_base else load_labeled()[0]
    new = pd.concat([load_raw_day(Path(p)) for p in a.new_data], ignore_index=True)
    rep = retrain(base, new, a.mode, a.holdout_frac, a.tolerance, a.promote)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
