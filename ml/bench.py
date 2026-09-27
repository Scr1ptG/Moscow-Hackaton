"""Замеры latency инференса (для раздела «производительность» формы).

Сравниваются продовая компактная модель и исследовательский ансамбль; CPU,
один поток. Запуск: ``python -m ml.bench`` -> ``reports/latency.json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .cache import ART_DIR, get_features, get_sequences
from .data import load_split
from .features import point_features, prepare_context
from .predictor import CompactPredictor, DelayPredictor
from .train import REPORT_DIR

MODEL_DIR = ART_DIR / "models"


def _timeit(fn, n: int) -> dict:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts[max(1, n // 10):])  # отбросить прогрев
    return {"p50_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def bench_online_cycle(pr, wide: bool, n_cycles: int = 8) -> dict:
    """Полный онлайн-цикл по всему парку test-дня на 09:00: контекст + признаки + прогноз + объяснение."""
    from .online import OnlineState

    sp = load_split("test")
    now0 = 1767690000.0
    tr = sp.traffic[sp.traffic["t"] <= now0 + 600]
    on = OnlineState(sp.schedule, wide=wide)
    on.add_records([dict(tr_id=int(r.tr_id), t=r.t, valid=bool(r.valid), lon=r.lon, lat=r.lat, speed=r.speed, heading=r.heading) for r in tr.itertuples()])

    def cycle(now):
        ctx = on.context(now)
        pts = on.prediction_points(now, ctx)
        X = pd.DataFrame([point_features(ctx, int(p.tr_id), now, int(p.target_stop_id), float(p.target_plan), float(p.cur_dev_s)) for p in pts.itertuples()])
        seq = None
        if pr.nn:
            from .nn import point_sequence

            seq = np.stack([point_sequence(ctx, int(p.tr_id), now, int(p.target_stop_id)) for p in pts.itertuples()])
        pr.predict_frame(X, seq)
        return len(pts)

    t0 = time.perf_counter()
    n = cycle(now0)
    cold = (time.perf_counter() - t0) * 1000
    warm = []
    for k in range(1, n_cycles + 1):
        t0 = time.perf_counter()
        cycle(now0 + 15 * k)
        warm.append((time.perf_counter() - t0) * 1000)
    return {"vehicles": n, "cold_ms": round(cold, 1), "warm_p50_ms": round(float(np.median(warm)), 1),
            "warm_per_vehicle_ms": round(float(np.median(warm)) / max(n, 1), 2)}


def bench_scale(copies: int = 50, shards: int = 1) -> dict:
    """Нагрузочный замер: реальный день, клонированный ``copies`` раз (новые tr_id),
    полный цикл прогноза по всему парку; при ``shards > 1`` — время на одну реплику."""
    from .online import OnlineState

    sp = load_split("test")
    now0 = 1767690000.0
    tr0 = sp.traffic[sp.traffic["t"] <= now0 + 600]
    plans, recs = [], []
    for c in range(copies):
        off = (c + 1) * 10_000_000
        plans.append(sp.schedule.assign(tr_id=sp.schedule["tr_id"] + off, stop_id=sp.schedule["stop_id"] + off * 100))
        recs += [dict(tr_id=int(r.tr_id) + off, t=r.t, valid=bool(r.valid), lon=r.lon, lat=r.lat, speed=r.speed, heading=r.heading) for r in tr0.itertuples()]
    plan = pd.concat(plans, ignore_index=True)
    pr = CompactPredictor.load()
    per_shard = []
    for i in range(shards):
        on = OnlineState(plan, wide=False, shard=(i, shards) if shards > 1 else None)
        on.add_records(recs)

        def cycle(now):
            ctx = on.context(now)
            pts = on.prediction_points(now, ctx)
            X = pd.DataFrame([point_features(ctx, int(p.tr_id), now, int(p.target_stop_id), float(p.target_plan), float(p.cur_dev_s)) for p in pts.itertuples()])
            pr.predict_frame(X)
            return len(pts)

        cycle(now0)  # холодный старт (детекция с начала дня)
        ts, n = [], 0
        for k in range(1, 4):
            t0 = time.perf_counter()
            n = cycle(now0 + 15 * k)
            ts.append((time.perf_counter() - t0) * 1000)
        per_shard.append({"vehicles": n, "warm_ms": round(float(np.median(ts)), 1)})
    total = sum(s["vehicles"] for s in per_shard)
    worst = max(s["warm_ms"] for s in per_shard)
    return {"copies": copies, "shards": shards, "active_vehicles": total, "per_shard": per_shard,
            "cycle_ms_parallel": worst, "ms_per_vehicle": round(worst * shards / max(total, 1), 2)}


def main(n: int = 200):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    sp = load_split("validate")
    ctx = prepare_context(sp)
    X = get_features("validate")
    p0 = sp.points.iloc[0]
    args = (int(p0.tr_id), float(p0["T"]), int(p0.target_stop_id), float(p0.target_plan), float(p0.cur_dev_s))
    res = {"features_1_point": _timeit(lambda: point_features(ctx, *args), n)}

    # ---- компактная продовая модель
    cp = CompactPredictor.load()
    x1 = X.iloc[[0]]
    xn = x1.reindex(columns=cp.feats).to_numpy(np.float64)
    res["compact"] = {
        "n_features": len(cp.feats),
        "model_1_point": _timeit(lambda: cp.model.predict(xn, num_threads=1), n),
        "shap_1_point": _timeit(lambda: cp.model.predict(xn, pred_contrib=True, num_threads=1), n),
        "predict_with_explanation_1_point": _timeit(lambda: cp.predict_frame(x1), n),
        "full_predict_point": _timeit(lambda: cp.predict_point(ctx, *args), max(30, n // 4)),
        "online_fleet_cycle": bench_online_cycle(cp, wide=False),
    }
    t0 = time.perf_counter()
    cp.predict_frame(X)
    res["compact"]["batch_151_points_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # ---- исследовательский ансамбль (для сравнения)
    if (MODEL_DIR / "meta.json").exists():
        ep = DelayPredictor.load()
        S = get_sequences("validate")
        xe = X.iloc[[0]].reindex(columns=ep.feats).to_numpy(np.float64)
        ens = {"n_features": len(ep.feats), "lgb_x5": _timeit(lambda: [m.predict(xe, num_threads=1) for m in ep.lgb], n)}
        if (MODEL_DIR / "cat_s0.cbm").exists():
            from catboost import CatBoostRegressor

            cb = CatBoostRegressor()
            cb.load_model(str(MODEL_DIR / "cat_s0.cbm"))
            ens["catboost_native_1"] = _timeit(lambda: cb.predict(xe, thread_count=1), n)
            if (MODEL_DIR / "cat_s0.onnx").exists():
                s = ep._ort_session(MODEL_DIR / "cat_s0.onnx")
                ens["catboost_onnx_1"] = _timeit(lambda: s.run(None, {"features": xe.astype(np.float32)}), n)
        if ep.nn:
            import torch

            from .nn import make_model, predict_nn

            xt = ep.nn_scaler.transform(X.iloc[[0]])
            s1 = S[[0]]
            tm = make_model(xt.shape[1])
            tm.load_state_dict(torch.load(MODEL_DIR / "nn_s0.pt", map_location="cpu"))
            tm.eval()
            ens["nn_torch_1"] = _timeit(lambda: predict_nn(tm, s1, xt), n)
            for name, f in (("nn_onnx_1", "nn_s0.onnx"), ("nn_onnx_int8_1", "nn_s0.int8.onnx")):
                if (MODEL_DIR / f).exists():
                    s = ep._ort_session(MODEL_DIR / f)
                    ens[name] = _timeit(lambda s=s: s.run(None, {"seq": s1.astype(np.float32), "tab": xt}), n)
        ens["full_predict_point"] = _timeit(lambda: ep.predict_point(ctx, *args), max(30, n // 4))
        ens["online_fleet_cycle"] = bench_online_cycle(ep, wide=True)
        res["ensemble"] = ens

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "latency.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
