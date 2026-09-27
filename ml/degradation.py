"""Надёжность: качество прогноза при обрыве связи (нет телеметрии последние L минут).

Оценка честная (leave-one-vehicle-out: модель не видела это ТС).
Для каждой размеченной точки признаки считаются в момент ``T``, но онлайн-контекст
содержит телеметрию только до ``T - L``. Так воспроизводится ситуация «связь с
ТС / эмулятором пропала L минут назад»: сервис не падает и прогнозирует по
последнему известному состоянию и плану.

Запуск: ``python -m ml.degradation`` -> ``reports/degradation.json``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .compact import FEATS, N_ROUNDS, REPORT_DIR, load_labeled, params, postprocess
from .data import load_split
from .features import point_features
from .models import mae
from .online import OnlineState
from .models import fit_lgb

LOSS_MIN = (0, 2, 5, 10, 20, 60)


def main():
    # честно: для каждого ТС — модель leave-one-vehicle-out, которая его не видела
    lab, _ = load_labeled()
    base_of = dict(zip(lab["tr_id"], lab["base_tr"]))
    fold_models = {g: fit_lgb(lab.loc[lab["base_tr"] != g, FEATS], lab.loc[lab["base_tr"] != g, "y"].to_numpy(), None,
                              params=params(), n_rounds=N_ROUNDS) for g in lab["base_tr"].unique()}
    rows = []
    for split in ("train", "test"):
        sp = load_split(split)
        pts = sp.points[~sp.points["is_synthetic"]]
        on = OnlineState(sp.schedule, wide=False)
        tr = sp.traffic[sp.traffic["tr_id"].isin(pts["tr_id"].unique())]
        on.add_records([dict(tr_id=int(r.tr_id), t=r.t, valid=bool(r.valid), lon=r.lon, lat=r.lat, speed=r.speed, heading=r.heading) for r in tr.itertuples()])
        for L in LOSS_MIN:
            feats = []
            for p in pts.itertuples():
                ctx = on.context(p.T - 60 * L, [int(p.tr_id)])
                feats.append(point_features(ctx, int(p.tr_id), float(p.T), int(p.target_stop_id), float(p.target_plan), float(p.cur_dev_s)))
            X = pd.DataFrame(feats)
            raw = np.array([fold_models[base_of[int(t)]].predict(X.iloc[[i]][FEATS])[0] for i, t in enumerate(pts["tr_id"])])
            pred = postprocess(raw, X["tgt_is_trip_start"].to_numpy())
            rows.append(pd.DataFrame({"L": L, "y": pts["y"].to_numpy(), "p": pred, "cur": pts["cur_dev_s"].to_numpy()}))
    d = pd.concat(rows)
    res = {f"{L} мин без телеметрии": round(mae(g["y"], g["p"]), 2) for L, g in d.groupby("L")}
    res["бейзлайн cur_dev_s (без телеметрии)"] = round(mae(d[d.L == 0]["y"], d[d.L == 0]["cur"]), 2)
    res["нулевой прогноз"] = round(mae(d[d.L == 0]["y"], 0), 2)
    res["n_points"] = int((d.L == 0).sum())
    (REPORT_DIR / "degradation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
