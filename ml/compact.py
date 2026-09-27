"""Компактная продовая модель: один LightGBM, 9 понятных признаков, строгая монотонность.

Почему так (эксперименты — ``REPORT.md``, раздел «Упрощение»):

* ансамбль LightGBM×5 + CatBoost×2 + GRU×3 на 86 признаках не лучше одной
  модели на 9 признаках (MAE 68.1 / 75.5 против 68.1 / 74.1, разница в шуме);
* одна модель даёт **точное** объяснение: прогноз = база + сумма TreeSHAP-вкладов;
* монотонные ограничения гарантируют физически осмысленное поведение
  (больше текущее опоздание -> не меньше прогноз), почти без потери качества.

Вероятность опоздания, интервал и ожидаемая ошибка получаются не отдельными
моделями, а из таблицы калибровки — распределения out-of-fold остатков по
корзинам прогноза. Это просто, прозрачно и откалибровано по построению.

Запуск::

    python -m ml.compact --cv --fit --submit --tag compact_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ART_DIR, get_all
from .data import base_vehicle_map, load_schedule_plan
from .models import fit_lgb, lgb_params, mae, score

ROOT = Path(__file__).resolve().parent.parent
COMPACT_DIR = ART_DIR / "compact"
REPORT_DIR = ROOT / "reports"

#: Признаки модели: имя -> (название для диспетчера, единица, монотонность)
FEATURES: dict[str, tuple[str, str, int]] = {
    "cur_dev_s": ("Текущее отклонение от графика (последняя плановая остановка)", "с", +1),
    "det_plus_hist_gain": ("Текущее отклонение + потери на этом участке в прошлых рейсах", "с", +1),
    "hist_tgt_delay": ("Отклонение на этой же остановке в прошлом рейсе", "с", +1),
    "det_stops_to_tgt": ("Остановок до целевой от последней пройденной", "шт", 0),
    "plan_path_speed": ("Плановая скорость на участке до цели (плотность графика)", "м/с", 0),
    "layover_slack_s": ("Запас времени на отстое на конечной", "с", -1),
    "speed_mean_5m": ("Средняя скорость за 5 мин", "км/ч", 0),
    "valid_share_30m": ("Доля валидных GPS-точек за 30 мин", "доля", 0),
    "hour": ("Время суток", "ч", 0),
}
FEATS = list(FEATURES)
MONOTONE = [FEATURES[f][2] for f in FEATS]

#: Группы для объяснения диспетчеру: код -> (текст, признаки)
GROUPS: dict[str, tuple[str, list[str]]] = {
    "current_delay": ("Текущее отклонение от графика", ["cur_dev_s"]),
    "segment_history": ("Этот участок в прошлых рейсах сегодня", ["det_plus_hist_gain", "hist_tgt_delay"]),
    "position": ("Сколько остановок осталось до цели", ["det_stops_to_tgt"]),
    "tight_schedule": ("Плотность планового графика", ["plan_path_speed"]),
    "terminal_layover": ("Отстой на конечной до целевой остановки", ["layover_slack_s"]),
    "traffic": ("Дорожная обстановка (скорость)", ["speed_mean_5m"]),
    "telematics": ("Качество телематики (GPS)", ["valid_share_30m"]),
    "time_of_day": ("Время суток", ["hour"]),
}

TRIP_START_ALPHA = 0.15
LATE_S = 120.0
N_ROUNDS = 300
PRED_BINS = [-np.inf, -30.0, 30.0, 90.0, 180.0, np.inf]


def params(seed: int = 0) -> dict:
    """Гиперпараметры: Huber (L1 несовместим с монотонностью), мелкие деревья."""
    return lgb_params(
        objective="huber",
        alpha=60.0,
        num_leaves=7,
        seed=seed,
        monotone_constraints=MONOTONE,
        monotone_constraints_method="intermediate",  # advanced давал редкие нарушения (до 1 с)
    )


def postprocess(pred: np.ndarray, trip_start: np.ndarray) -> np.ndarray:
    """Правило «старт рейса»: ТС выпускают по расписанию, прогноз прижимается к 0."""
    return np.where(np.asarray(trip_start) == 1, TRIP_START_ALPHA * pred, pred)


def load_labeled() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_all()
    bm = base_vehicle_map(load_schedule_plan("train"))
    df["base_tr"] = df["tr_id"].map(bm).fillna(df["tr_id"]).astype(int)
    lab = df[df["split"] != "validate"].reset_index(drop=True)
    val = df[df["split"] == "validate"].reset_index(drop=True)
    return lab, val


def lovo_oof(lab: pd.DataFrame) -> np.ndarray:
    """Out-of-fold прогноз leave-one-vehicle-out (с пост-обработкой)."""
    oof = np.full(len(lab), np.nan)
    real = ~lab["is_synthetic"].to_numpy()
    for g in sorted(lab["base_tr"].unique()):
        te = (lab["base_tr"] == g).to_numpy() & real
        tr = (lab["base_tr"] != g).to_numpy()
        m = fit_lgb(lab.loc[tr, FEATS], lab["y"].to_numpy()[tr], None, params=params(), n_rounds=N_ROUNDS)
        oof[te] = m.predict(lab.loc[te, FEATS])
    return postprocess(oof, lab["tgt_is_trip_start"].to_numpy())


def calibration_table(pred: np.ndarray, y: np.ndarray) -> dict:
    """Остатки out-of-fold по корзинам прогноза -> P(опоздание), интервал, ожидаемая абсолютная ошибка."""
    resid = y - pred
    bins = np.digitize(pred, PRED_BINS[1:-1])
    table = []
    for b in range(len(PRED_BINS) - 1):
        r = resid[bins == b]
        table.append({
            "lo": PRED_BINS[b], "hi": PRED_BINS[b + 1], "n": int(len(r)),
            "resid": np.round(np.quantile(r, np.linspace(0, 1, 101)), 2).tolist(),  # 101 квантиль остатка
            "mean_abs": float(np.mean(np.abs(r))),
        })
    return {"bins": PRED_BINS, "table": table}


def calibrated(pred: np.ndarray, calib: dict) -> dict[str, np.ndarray]:
    """P(y > 120 c), интервал q10–q90 и ожидаемая абсолютная ошибка по таблице калибровки."""
    pred = np.asarray(pred, dtype=float)
    bins = np.digitize(pred, np.array(calib["bins"][1:-1], dtype=float))
    p_late, q10, q90, eae = (np.empty(len(pred)) for _ in range(4))
    for i, (p, b) in enumerate(zip(pred, bins)):
        row = calib["table"][b]
        r = np.asarray(row["resid"])
        p_late[i] = float(np.mean(p + r > LATE_S))
        q10[i], q90[i] = p + r[10], p + r[90]
        eae[i] = row["mean_abs"]
    return {"p_late": p_late, "q10": q10, "q90": q90, "expected_abs_error": eae}


def cv_report(lab: pd.DataFrame) -> dict:
    """CV-отчёт компактной модели + сравнение с ансамблем (``reports/cv_oof.csv``)."""
    oof = lovo_oof(lab)
    real = ~lab["is_synthetic"].to_numpy()
    y = lab["y"].to_numpy()
    te = (lab["split"] == "test").to_numpy()
    d = lab[real].assign(p=oof[real])

    def seg(m):
        dd = d[m]
        return {"n": int(len(dd)), "mae_zero": mae(dd["y"], 0), "mae_cur_dev": mae(dd["y"], dd["cur_dev_s"]), "mae_model": mae(dd["y"], dd["p"])}

    rep = {
        "model": "LightGBM Huber, 7 листьев x 300 деревьев, строгая монотонность, 9 признаков + правило старта рейса",
        "features": FEATS,
        "mae_real": mae(y[real], oof[real]),
        "mae_test": mae(y[te], oof[te]),
        "score_test_est": score(y[te], oof[te]),
        "segments": {
            "все реальные": seg(np.ones(len(d), bool)),
            "только test": seg((d["split"] == "test").to_numpy()),
            "без отстоя на пути": seg((d["n_trip_starts_path"] == 0).to_numpy()),
            "с отстоем на пути": seg((d["n_trip_starts_path"] > 0).to_numpy()),
            "цель = старт рейса": seg((d["tgt_is_trip_start"] == 1).to_numpy()),
            "GPS деградирован (valid_15m<0.5)": seg((d["valid_share_15m"].fillna(0) < 0.5).to_numpy()),
        },
    }
    ens = REPORT_DIR / "cv_oof.csv"
    if ens.exists():
        e = pd.read_csv(ens).set_index("sample_id")["oof"]
        pe = postprocess(d["sample_id"].map(e).to_numpy(), d["tgt_is_trip_start"].to_numpy())
        rep["ensemble_same_points"] = {"mae_real": mae(d["y"], pe), "mae_test": mae(d["y"][d["split"] == "test"], pe[(d["split"] == "test").to_numpy()])}
    calib = calibration_table(d["p"].to_numpy(), d["y"].to_numpy())
    c = calibrated(d["p"].to_numpy(), calib)
    late = (d["y"].to_numpy() > LATE_S).astype(float)
    rep["calibration_check"] = {
        "interval_q10_q90_coverage": float(np.mean((d["y"] >= c["q10"]) & (d["y"] <= c["q90"]))),
        "brier_p_late": float(np.mean((c["p_late"] - late) ** 2)),
        "brier_constant": float(np.mean((late.mean() - late) ** 2)),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "cv_report_compact.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    d[["sample_id", "tr_id", "split", "y", "cur_dev_s", "p"]].rename(columns={"p": "oof"}).to_csv(REPORT_DIR / "cv_oof_compact.csv", index=False)
    return rep, calib


def fit(lab: pd.DataFrame, calib: dict, protected: pd.DataFrame | None = None) -> dict:
    """Финальная модель + таблица калибровки + описание признаков.

    :param protected: прогнозные точки, ответы на которые модель видеть не должна (validate).
        Синтетические копии их периодов (±30 мин) исключаются из обучения — иначе модель
        «подсматривает» ответ через копию (см. ``ml.protocol.twin_contaminated``).
    """
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    n_removed = 0
    if protected is not None:
        from .protocol import twin_contaminated

        bad = twin_contaminated(lab, protected)
        n_removed = int(bad.sum())
        lab = lab[~bad]
    m = fit_lgb(lab[FEATS], lab["y"].to_numpy(), None, params=params(), n_rounds=N_ROUNDS)
    m.save_model(str(COMPACT_DIR / "model.txt"))
    meta = {
        "kind": "compact",
        "features": FEATS,
        "feature_info": {f: {"title": t, "unit": u, "monotone": mo} for f, (t, u, mo) in FEATURES.items()},
        "groups": {k: {"title": t, "features": fs} for k, (t, fs) in GROUPS.items()},
        "postprocess": {"trip_start_alpha": TRIP_START_ALPHA},
        "late_s": LATE_S,
        "calibration": calib,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": int(len(lab)),
        "training_policy": "реальные ТС + синтетика без копий защищённых периодов (±30 мин)" if protected is not None else "вся разметка",
        "synthetic_twins_removed": n_removed,
    }
    (COMPACT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="compact")
    ap.add_argument("--allow-twins", action="store_true", help="не исключать копии validate-периодов (нечестно)")
    a = ap.parse_args()
    lab, val = load_labeled()
    rep, calib = cv_report(lab) if (a.cv or a.fit) else (None, None)
    if rep:
        print(json.dumps({k: rep[k] for k in ("mae_real", "mae_test", "score_test_est", "calibration_check")}, ensure_ascii=False, indent=2))
        if "ensemble_same_points" in rep:
            print("ансамбль на тех же точках:", rep["ensemble_same_points"])
    if a.fit:
        meta = fit(lab, calib, protected=None if a.allow_twins else val)
        print("[fit] ->", COMPACT_DIR, "| исключено копий validate:", meta["synthetic_twins_removed"])
    if a.submit:
        from .predictor import CompactPredictor
        from .train import write_submission

        pr = CompactPredictor.load()
        out = pr.predict_frame(val, explain=False)
        path = write_submission(pd.Series(out["delay_pred"].to_numpy(), index=val["sample_id"]), a.tag)
        print("[submit] ->", path)


if __name__ == "__main__":
    main()
