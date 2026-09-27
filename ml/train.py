"""Обучение финального ансамбля, CV-отчёт и формирование ``submission.csv``.

Запуск::

    python -m ml.train --cv          # leave-one-vehicle-out CV + отчёт в reports/
    python -m ml.train --fit --submit  # финальные модели + submission по validate

Ансамбль: LightGBM (L1, несколько сидов) + CatBoost (MAE) [+ GRU/MLP, если
включена], веса подбираются по out-of-fold MAE. Дополнительно обучаются
вспомогательные модели для дашборда: квантили q10/q90 (интервал прогноза) и
классификатор P(опоздание > 120 c).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ART_DIR, get_all, get_all_sequences
from .data import DATA_DIR, base_vehicle_map, load_schedule_plan
from .models import feature_columns, fit_cat, fit_lgb, lgb_params, mae, score

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ART_DIR / "models"
REPORT_DIR = ROOT / "reports"
SUB_DIR = ROOT / "submissions"

LATE_S = 120.0  # порог «опоздания» (как target_class в разметке)
EARLY_S = -60.0
#: Сжатие прогноза для целей «старт рейса» (вложенная проверка по ТС: -0.89 MAE, 20/20 разбиений)
TRIP_START_ALPHA = 0.15

CONFIG = {
    "lgb_seeds": [0, 1, 2, 3, 4],
    "lgb_rounds": 500,
    "cat_seeds": [0, 1],
    "cat_iters": 1000,
    "nn_seeds": [0, 1, 2],
    "nn_epochs": 60,
    "use_nn": False,
    "syn_weight": 1.0,
}


# ------------------------------------------------------------------ данные


def load_table():
    """Признаки всех сплитов + базовое ТС (для групп CV) + окна для NN."""
    df = get_all()
    bm = base_vehicle_map(load_schedule_plan("train"))
    df["base_tr"] = df["tr_id"].map(bm).fillna(df["tr_id"]).astype(int)
    seq = get_all_sequences() if CONFIG["use_nn"] else None
    return df, seq


# ------------------------------------------------------------------ модели


def _fit_predict_family(name, tr_df, te_df, feats, y, w, seq_tr=None, seq_te=None):
    """Обучает семейство моделей (все сиды) и возвращает средний прогноз."""
    if name == "lgb":
        ms = [fit_lgb(tr_df[feats], y, w, params=lgb_params(seed=s), n_rounds=CONFIG["lgb_rounds"]) for s in CONFIG["lgb_seeds"]]
        return np.mean([m.predict(te_df[feats]) for m in ms], axis=0), ms
    if name == "cat":
        ms = [fit_cat(tr_df[feats], y, w, params=dict(iterations=CONFIG["cat_iters"], random_seed=s)) for s in CONFIG["cat_seeds"]]
        return np.mean([m.predict(te_df[feats]) for m in ms], axis=0), ms
    if name == "nn":
        from .nn import TabScaler, fit_nn, predict_nn

        sc = TabScaler.fit(tr_df, feats)
        xtr, xte = sc.transform(tr_df), sc.transform(te_df)
        ms = [fit_nn(seq_tr, xtr, y.astype("float32"), w, epochs=CONFIG["nn_epochs"], seed=s) for s in CONFIG["nn_seeds"]]
        return np.mean([predict_nn(m, seq_te, xte) for m in ms], axis=0), (sc, ms)
    raise ValueError(name)


def families() -> list[str]:
    return ["lgb", "cat"] + (["nn"] if CONFIG["use_nn"] else [])


def best_blend(oofs: dict[str, np.ndarray], y: np.ndarray, step: float = 0.05) -> dict[str, float]:
    """Веса смеси на симплексе, минимизирующие OOF MAE (перебор по сетке)."""
    names = list(oofs)
    grid = np.arange(0, 1 + 1e-9, step)
    best, best_w = np.inf, None
    for ws in itertools.product(grid, repeat=len(names) - 1):
        if sum(ws) > 1 + 1e-9:
            continue
        w = list(ws) + [1 - sum(ws)]
        p = sum(wi * oofs[n] for wi, n in zip(w, names))
        m = mae(y, p)
        if m < best:
            best, best_w = m, w
    return {n: round(float(wi), 3) for n, wi in zip(names, best_w)}


# ------------------------------------------------------------------ CV


def run_cv_report(df: pd.DataFrame, seq) -> dict:
    """Leave-one-vehicle-out CV всех семейств, подбор весов, отчёт по сегментам."""
    lab_mask = (df["split"] != "validate").to_numpy()
    lab = df[lab_mask].reset_index(drop=True)
    lseq = seq[lab_mask] if seq is not None else None
    feats = feature_columns(lab)
    y = lab["y"].to_numpy()
    w = np.where(lab["is_synthetic"], CONFIG["syn_weight"], 1.0)
    real = ~lab["is_synthetic"].to_numpy()
    oofs = {n: np.full(len(lab), np.nan) for n in families()}
    t0 = time.time()
    for g in sorted(lab["base_tr"].unique()):
        te = (lab["base_tr"] == g).to_numpy() & real
        tr = (lab["base_tr"] != g).to_numpy()
        if te.sum() == 0:
            continue
        for n in families():
            p, _ = _fit_predict_family(
                n, lab[tr], lab[te], feats, y[tr], w[tr],
                lseq[tr] if lseq is not None else None, lseq[te] if lseq is not None else None,
            )
            oofs[n][te] = p
        print(f"[cv] fold base_tr={g} готов ({time.time() - t0:.0f} c)")
    yr = y[real]
    blend_w = best_blend({n: o[real] for n, o in oofs.items()}, yr)
    blend = sum(blend_w[n] * oofs[n] for n in oofs)
    te_mask = (lab["split"] == "test").to_numpy()

    def seg_table(pred):
        rows = {}
        d = lab[real].assign(p=pred[real])
        for name, m in {
            "все реальные": np.ones(len(d), bool),
            "только test": (d["split"] == "test").to_numpy(),
            "без отстоя на пути": (d["n_trip_starts_path"] == 0).to_numpy(),
            "с отстоем на пути": (d["n_trip_starts_path"] > 0).to_numpy(),
            "цель = старт рейса": (d["tgt_is_trip_start"] == 1).to_numpy(),
            "GPS деградирован (valid_15m<0.5)": (d["valid_share_15m"].fillna(0) < 0.5).to_numpy(),
        }.items():
            dd = d[m]
            rows[name] = {
                "n": int(len(dd)),
                "mae_zero": mae(dd["y"], 0),
                "mae_cur_dev": mae(dd["y"], dd["cur_dev_s"]),
                "mae_model": mae(dd["y"], dd["p"]),
            }
        return rows

    rep = {
        "features": feats,
        "n_features": len(feats),
        "cv": "leave-one-vehicle-out (реальное ТС + его синтетические копии в одном фолде), метрика по реальным ТС",
        "families": {
            n: {"mae_real": mae(yr, o[real]), "mae_test": mae(y[te_mask], o[te_mask])} for n, o in oofs.items()
        },
        "blend_weights": blend_w,
        "blend": {
            "mae_real": mae(yr, blend[real]),
            "mae_test": mae(y[te_mask], blend[te_mask]),
            "score_test_est": score(y[te_mask], blend[te_mask]),
            "score_real_est": score(yr, blend[real]),
        },
        "baselines": {
            "zero": {"mae_real": mae(yr, 0), "mae_test": mae(y[te_mask], 0)},
            "cur_dev_s": {"mae_real": mae(yr, lab["cur_dev_s"].to_numpy()[real]), "mae_test": mae(y[te_mask], lab["cur_dev_s"].to_numpy()[te_mask]),
                          "score_test_est": score(y[te_mask], lab["cur_dev_s"].to_numpy()[te_mask])},
        },
        "segments": seg_table(blend),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "cv_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    lab.assign(oof=blend)[["sample_id", "tr_id", "split", "is_synthetic", "y", "cur_dev_s", "oof"]].to_csv(REPORT_DIR / "cv_oof.csv", index=False)
    return rep


# ------------------------------------------------------------------ финальное обучение


def fit_final(df: pd.DataFrame, seq, blend_w: dict[str, float]) -> dict:
    """Обучает финальные модели на всей разметке (train+test) и сохраняет артефакты."""
    import lightgbm as lgb

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lab_mask = (df["split"] != "validate").to_numpy()
    lab = df[lab_mask].reset_index(drop=True)
    feats = feature_columns(lab)
    y = lab["y"].to_numpy()
    w = np.where(lab["is_synthetic"], CONFIG["syn_weight"], 1.0)
    meta = {
        "features": feats,
        "blend_weights": blend_w,
        "config": CONFIG,
        "models": {},
        "postprocess": {"trip_start_alpha": TRIP_START_ALPHA},
    }
    for s in CONFIG["lgb_seeds"]:
        m = fit_lgb(lab[feats], y, w, params=lgb_params(seed=s), n_rounds=CONFIG["lgb_rounds"])
        m.save_model(str(MODEL_DIR / f"lgb_s{s}.txt"))
        meta["models"].setdefault("lgb", []).append(f"lgb_s{s}.txt")
    if blend_w.get("cat", 0) > 0:
        for s in CONFIG["cat_seeds"]:
            m = fit_cat(lab[feats], y, w, params=dict(iterations=CONFIG["cat_iters"], random_seed=s))
            m.save_model(str(MODEL_DIR / f"cat_s{s}.cbm"))
            try:  # ONNX-версия для быстрого инференса без зависимостей catboost
                m.save_model(str(MODEL_DIR / f"cat_s{s}.onnx"), format="onnx")
            except Exception as e:  # noqa: BLE001
                print("[fit] ONNX-экспорт CatBoost не удался:", e)
            meta["models"].setdefault("cat", []).append(f"cat_s{s}.cbm")
    if CONFIG["use_nn"] and blend_w.get("nn", 0) > 0:
        import torch

        from .nn import TabScaler, fit_nn

        lseq = seq[lab_mask]
        sc = TabScaler.fit(lab, feats)
        x = sc.transform(lab)
        json.dump({"cols": sc.cols, "med": sc.med.tolist(), "iqr": sc.iqr.tolist()}, open(MODEL_DIR / "nn_scaler.json", "w"))
        for s in CONFIG["nn_seeds"]:
            m = fit_nn(lseq, x, y.astype("float32"), w, epochs=CONFIG["nn_epochs"], seed=s)
            torch.save(m.state_dict(), MODEL_DIR / f"nn_s{s}.pt")
            meta["models"].setdefault("nn", []).append(f"nn_s{s}.pt")
    # вспомогательные модели для дашборда
    for name, params in {
        "q10": lgb_params(objective="quantile", alpha=0.1),
        "q90": lgb_params(objective="quantile", alpha=0.9),
    }.items():
        fit_lgb(lab[feats], y, w, params=params, n_rounds=400).save_model(str(MODEL_DIR / f"lgb_{name}.txt"))
    for name, thr in {"p_late": LATE_S}.items():
        clf = lgb.train(
            lgb_params(objective="binary", metric="binary_logloss"),
            lgb.Dataset(lab[feats], (y > thr).astype(int), weight=w),
            num_boost_round=300,
        )
        clf.save_model(str(MODEL_DIR / f"lgb_{name}.txt"))
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def fit_error_model(df: pd.DataFrame) -> dict:
    """Модель ожидаемой абсолютной ошибки прогноза ``E|y - ŷ|`` по OOF-остаткам.

    Учится на out-of-fold прогнозах ансамбля (``reports/cv_oof.csv``), поэтому
    оценивает ошибку честно — как на невиданных данных. Используется
    дашбордом как «ожидаемая точность» конкретного прогноза.
    """
    oof = pd.read_csv(REPORT_DIR / "cv_oof.csv")
    lab = df[df["split"] != "validate"].merge(oof[["sample_id", "oof"]], on="sample_id")
    lab = lab[lab["oof"].notna()]
    feats = feature_columns(lab.drop(columns=["oof"]))
    X = lab[feats].assign(delay_pred=lab["oof"])
    err = np.abs(lab["y"] - lab["oof"])
    m = fit_lgb(X, err, None, params=lgb_params(objective="l1", num_leaves=7, min_data_in_leaf=60), n_rounds=300)
    m.save_model(str(MODEL_DIR / "lgb_abs_err.txt"))
    pred = m.predict(X)
    return {"n": int(len(lab)), "mean_abs_err": float(err.mean()), "in_sample_corr": float(np.corrcoef(pred, err)[0, 1])}


def export_nn_onnx() -> list[str]:
    """Экспорт обученных NN в ONNX + динамическая int8-квантизация (ONNX Runtime)."""
    import torch

    from .nn import export_onnx, make_model

    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    sc = json.load(open(MODEL_DIR / "nn_scaler.json"))
    n_tab = 2 * len(sc["cols"])
    out = []
    for f in meta["models"].get("nn", []):
        m = make_model(n_tab)
        m.load_state_dict(torch.load(MODEL_DIR / f, map_location="cpu"))
        path = MODEL_DIR / f.replace(".pt", ".onnx")
        export_onnx(m, n_tab, path)
        out.append(path.name)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            q = MODEL_DIR / f.replace(".pt", ".int8.onnx")
            quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
            out.append(q.name)
        except Exception as e:  # noqa: BLE001
            print("[onnx] квантизация не удалась:", e)
    return out


# ------------------------------------------------------------------ submission


def write_submission(pred: pd.Series, tag: str) -> Path:
    """Пишет ``sample_id;prediction`` и проверяет формат против sample_submission."""
    tmpl = pd.read_csv(DATA_DIR / "sample_submission.csv", sep=";")
    sub = tmpl[["sample_id"]].copy()
    sub["prediction"] = sub["sample_id"].map(pred)
    assert sub["prediction"].notna().all(), "есть sample_id без прогноза"
    assert sub["sample_id"].is_unique and len(sub) == len(tmpl)
    assert np.isfinite(sub["prediction"]).all()
    sub["prediction"] = sub["prediction"].round(1)
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"submission_{tag}.csv"
    sub.to_csv(path, sep=";", index=False, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--nn", action="store_true", help="включить GRU/MLP в ансамбль")
    ap.add_argument("--onnx", action="store_true", help="экспорт NN в ONNX (+int8)")
    ap.add_argument("--errmodel", action="store_true", help="модель ожидаемой абсолютной ошибки по OOF")
    ap.add_argument("--tag", default=time.strftime("%m%d_%H%M"))
    a = ap.parse_args()
    CONFIG["use_nn"] = a.nn
    df, seq = load_table()
    blend_w = {"lgb": 0.6, "cat": 0.4}
    if a.cv:
        rep = run_cv_report(df, seq)
        blend_w = rep["blend_weights"]
        print(json.dumps({k: rep[k] for k in ("families", "blend_weights", "blend", "baselines")}, ensure_ascii=False, indent=2))
    elif (REPORT_DIR / "cv_report.json").exists():
        blend_w = json.loads((REPORT_DIR / "cv_report.json").read_text(encoding="utf-8"))["blend_weights"]
    if a.fit:
        fit_final(df, seq, blend_w)
        print("[fit] модели сохранены в", MODEL_DIR)
    if a.onnx:
        print("[onnx]", export_nn_onnx())
    if a.errmodel:
        print("[errmodel]", fit_error_model(df))
    if a.submit:
        from .predictor import DelayPredictor

        pr = DelayPredictor.load()
        val = df[df["split"] == "validate"].reset_index(drop=True)
        vseq = seq[(df["split"] == "validate").to_numpy()] if seq is not None else None
        p = pr.predict_frame(val, vseq)
        path = write_submission(pd.Series(p["delay_pred"].to_numpy(), index=val["sample_id"]), a.tag)
        print("[submit] ->", path, "| mean pred", round(float(p["delay_pred"].mean()), 1))


if __name__ == "__main__":
    main()
