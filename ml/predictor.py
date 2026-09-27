"""Рантайм-инференс: загрузка артефактов и прогноз (офлайн-батч и онлайн-точка).

Две модели с одинаковым интерфейсом:

* :class:`CompactPredictor` — **продовая**: один LightGBM на 12 признаках,
  точное аддитивное объяснение (база + вклады = прогноз), калибровка по OOF;
* :class:`DelayPredictor` — исследовательский ансамбль (LightGBM+CatBoost+GRU, 86 признаков).

Выход для каждой прогнозной точки:

* ``delay_pred`` — прогноз задержки на целевой остановке, с;
* ``q10`` / ``q90`` — интервал прогноза;
* ``p_late`` — вероятность опоздания > 120 с;
* ``expected_abs_error`` — ожидаемая абсолютная ошибка прогноза, с;
* ``risk`` — уровень риска для дашборда: ``green`` / ``yellow`` / ``red``;
* ``cause_code`` / ``cause`` — главная причина (группа с наибольшим вкладом);
* ``explanation`` — текст для карточки инцидента (компактная модель);
* ``patterns`` — правила-паттерны поведения ТС перед сбоем.

:func:`load_predictor` выбирает модель по переменной окружения ``ML_MODEL``
(``compact`` по умолчанию, ``ensemble`` — ансамбль).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ART_DIR
from .explain import FACTOR_GROUPS, behaviour_patterns, group_contributions, recommendations, top_cause

MODEL_DIR = ART_DIR / "models"


def risk_level(delay: float, p_late: float) -> str:
    """Светофор риска для диспетчера (пороги как у классов разметки: -60 / +120 с)."""
    if delay >= 180 or p_late >= 0.6:
        return "red"
    if delay >= 60 or p_late >= 0.3 or delay <= -90:
        return "yellow"
    return "green"


class DelayPredictor:
    """Ансамбль LightGBM + CatBoost (+ GRU/MLP) и вспомогательные модели."""

    needs_wide = True  # использует признаки «широкой» детекции (радиус 30 м)

    def __init__(self, meta: dict, model_dir: Path):
        import lightgbm as lgb

        self.meta = meta
        self.feats: list[str] = meta["features"]
        self.w: dict[str, float] = meta["blend_weights"]
        # онлайн-инференс идёт по одной точке: однопоточный режим быстрее и не даёт
        # OpenMP (LightGBM) и пулу потоков ONNX Runtime конкурировать (иначе x30 latency)
        self.n_threads = int(os.environ.get("ML_THREADS", "1"))
        self.lgb = [lgb.Booster(model_file=str(model_dir / f)) for f in meta["models"].get("lgb", [])]
        backend = os.environ.get("ML_NN_BACKEND", "onnx")
        self.cat = []
        self.cat_backend = None
        if self.w.get("cat", 0) > 0 and meta["models"].get("cat"):
            cat_onnx = [model_dir / f.replace(".cbm", ".onnx") for f in meta["models"]["cat"]]
            if backend == "onnx" and all(p.exists() for p in cat_onnx):
                # ONNX-версия CatBoost: те же прогнозы (|diff| < 1e-4), но ~0.01 мс на точку
                self.cat = [self._ort_session(p) for p in cat_onnx]
                self.cat_backend = "onnx"
            else:
                from catboost import CatBoostRegressor

                for f in meta["models"]["cat"]:
                    m = CatBoostRegressor()
                    m.load_model(str(model_dir / f))
                    self.cat.append(m)
                self.cat_backend = "native"
        self.nn = []
        self.nn_backend = None
        self.nn_scaler = None
        if self.w.get("nn", 0) > 0 and meta["models"].get("nn"):
            from .nn import TabScaler

            sc = json.load(open(model_dir / "nn_scaler.json"))
            self.nn_scaler = TabScaler(sc["cols"], np.array(sc["med"]), np.array(sc["iqr"]))
            onnx_files = [model_dir / f.replace(".pt", ".onnx") for f in meta["models"]["nn"]]
            if backend == "onnx" and all(p.exists() for p in onnx_files):
                self.nn = [self._ort_session(p) for p in onnx_files]
                self.nn_backend = "onnx"
            else:
                import torch

                from .nn import make_model

                for f in meta["models"]["nn"]:
                    m = make_model(2 * len(sc["cols"]))
                    m.load_state_dict(torch.load(model_dir / f, map_location="cpu"))
                    m.eval()
                    self.nn.append(m)
                self.nn_backend = "torch"
        self.q10 = lgb.Booster(model_file=str(model_dir / "lgb_q10.txt"))
        self.q90 = lgb.Booster(model_file=str(model_dir / "lgb_q90.txt"))
        self.p_late = lgb.Booster(model_file=str(model_dir / "lgb_p_late.txt"))
        err_path = model_dir / "lgb_abs_err.txt"
        self.abs_err = lgb.Booster(model_file=str(err_path)) if err_path.exists() else None

    def _ort_session(self, path: Path):
        """Сессия ONNX Runtime без spin-wait (иначе конкурирует с OpenMP LightGBM)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.n_threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.log_severity_level = 3
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "DelayPredictor":
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(meta, model_dir)

    # -------------------------------------------------------------- батч
    def predict_frame(self, X: pd.DataFrame, seq: np.ndarray | None = None, explain: bool = True) -> pd.DataFrame:
        """Прогноз для таблицы признаков (колонки — как в обучении)."""
        X = X.reindex(columns=self.feats)
        Xn = X.to_numpy(dtype=np.float64)
        nt = self.n_threads
        parts = {"lgb": np.mean([m.predict(Xn, num_threads=nt) for m in self.lgb], axis=0)}
        if self.cat:
            if self.cat_backend == "onnx":
                x32 = Xn.astype(np.float32)
                parts["cat"] = np.mean([m.run(None, {"features": x32})[0].ravel() for m in self.cat], axis=0)
            else:
                parts["cat"] = np.mean([m.predict(Xn, thread_count=nt) for m in self.cat], axis=0)
        if self.nn and seq is not None:
            xt = self.nn_scaler.transform(X)
            if self.nn_backend == "onnx":
                s32 = np.asarray(seq, dtype=np.float32)
                parts["nn"] = np.mean([m.run(None, {"seq": s32, "tab": xt})[0] for m in self.nn], axis=0)
            else:
                from .nn import predict_nn

                parts["nn"] = np.mean([predict_nn(m, seq, xt) for m in self.nn], axis=0)
        wsum = sum(self.w.get(k, 0) for k in parts)
        pred = sum(self.w.get(k, 0) * v for k, v in parts.items()) / wsum
        # пост-обработка: на старте рейса ТС выпускают по расписанию, и факт там —
        # шум около нуля; модель в этом сегменте хуже нуля, поэтому прогноз
        # сжимается к 0 (коэффициент подобран вложенной проверкой по ТС)
        alpha = self.meta.get("postprocess", {}).get("trip_start_alpha")
        if alpha is not None and "tgt_is_trip_start" in X:
            ts = (X["tgt_is_trip_start"] == 1).to_numpy()
            pred = np.where(ts, alpha * pred, pred)
        out = pd.DataFrame(
            {
                "delay_pred": pred,
                "q10": self.q10.predict(Xn, num_threads=nt),
                "q90": self.q90.predict(Xn, num_threads=nt),
                "p_late": self.p_late.predict(Xn, num_threads=nt),
            }
        )
        out["q10"] = np.minimum(out["q10"], out["delay_pred"])
        out["q90"] = np.maximum(out["q90"], out["delay_pred"])
        if self.abs_err is not None:
            Xe = np.column_stack([Xn, out["delay_pred"].to_numpy()])
            out["expected_abs_error"] = np.maximum(self.abs_err.predict(Xe, num_threads=nt), 0)
        out["risk"] = [risk_level(d, p) for d, p in zip(out["delay_pred"], out["p_late"])]
        if explain:
            contrib = self.lgb[0].predict(Xn, pred_contrib=True, num_threads=nt)
            groups = group_contributions(contrib, self.feats)
            causes = [top_cause(groups.iloc[i], out["delay_pred"].iat[i]) for i in range(len(out))]
            out["cause_code"] = [c[0] for c in causes]
            out["cause"] = [c[1] for c in causes]
            out["patterns"] = [behaviour_patterns(r) for r in X.to_dict("records")]
            for code in FACTOR_GROUPS:
                out[f"contrib_{code}"] = groups[code].to_numpy()
        return out

    # -------------------------------------------------------------- онлайн
    def predict_point(self, ctx, tr_id: int, T: float, target_stop_id: int, target_plan: float, cur_dev_s: float) -> dict:
        """Онлайн-прогноз одной точки по контексту (треки и план в памяти backend)."""
        from .features import point_features

        f = point_features(ctx, tr_id, T, target_stop_id, target_plan, cur_dev_s)
        seq = None
        if self.nn:
            from .nn import point_sequence

            seq = point_sequence(ctx, tr_id, T, target_stop_id)[None]
        row = self.predict_frame(pd.DataFrame([f]), seq).iloc[0].to_dict()
        row["features"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items()}
        return row


# ============================================================ компактная модель


def fmt_signed(sec: float) -> str:
    """Секунды -> «+2:05» / «−0:40»."""
    s = "+" if sec >= 0 else "−"
    a = int(round(abs(sec)))
    return f"{s}{a // 60}:{a % 60:02d}"


def fmt_value(v: float, unit: str) -> str:
    """Значение признака в читаемом для диспетчера виде."""
    if v is None or not np.isfinite(v):
        return "нет данных"
    if unit == "с":
        return f"{fmt_signed(v)} (мин:с)"
    if unit == "доля":
        return f"{v:.0%}"
    if unit == "ч":
        return f"{int(v) % 24:02d}:{int(round((v % 1) * 60)) % 60:02d}"
    if unit == "шт":
        return f"{int(round(v))}"
    if unit == "с/с":
        return f"{v:+.2f} с за с плана"
    return f"{v:.1f} {unit}"


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


class CompactPredictor:
    """Продовая модель: один LightGBM, 12 признаков, точное объяснение.

    Объяснение аддитивно и точно:
    ``delay_pred = base_value + сумма вкладов групп + rule_contrib``, где вклады —
    TreeSHAP одной модели, а ``rule_contrib`` — поправка правила «старт рейса».
    """

    nn: list = []  # совместимость с интерфейсом DelayPredictor (нет последовательностей)
    needs_wide = False  # 11 признаков не требуют «широкой» детекции -> онлайн быстрее

    def __init__(self, meta: dict, model_dir: Path):
        import lightgbm as lgb

        self.meta = meta
        self.feats: list[str] = meta["features"]
        self.groups: dict = meta["groups"]
        self.info: dict = meta["feature_info"]
        self.calib: dict = meta["calibration"]
        self.alpha: float = meta["postprocess"]["trip_start_alpha"]
        self.model = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.n_threads = int(os.environ.get("ML_THREADS", "1"))
        idx = {f: i for i, f in enumerate(self.feats)}
        self._gidx = {g: [idx[f] for f in v["features"]] for g, v in self.groups.items()}
        # таблица калибровки -> numpy один раз (горячий путь без конвертаций)
        self._cal_edges = np.asarray(self.calib["bins"][1:-1], dtype=float)
        self._cal_resid = [np.asarray(r["resid"], dtype=float) for r in self.calib["table"]]
        self._cal_mae = np.asarray([r["mean_abs"] for r in self.calib["table"]], dtype=float)
        self._late = float(meta.get("late_s", 120.0))

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "CompactPredictor":
        from .compact import COMPACT_DIR

        model_dir = model_dir or COMPACT_DIR
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(meta, model_dir)

    def _calibrate(self, pred: np.ndarray) -> dict[str, np.ndarray]:
        """P(опоздание), интервал q10–q90 и ожидаемая абсолютная ошибка (векторно по корзинам)."""
        b = np.digitize(pred, self._cal_edges)
        p_late, q10, q90 = np.empty(len(pred)), np.empty(len(pred)), np.empty(len(pred))
        for k in np.unique(b):
            m = b == k
            r = self._cal_resid[k]
            p = pred[m]
            p_late[m] = (p[:, None] + r[None, :] > self._late).mean(axis=1)
            q10[m], q90[m] = p + r[10], p + r[90]
        return {"q10": np.minimum(q10, pred), "q90": np.maximum(q90, pred), "p_late": p_late, "expected_abs_error": self._cal_mae[b]}

    def _core(self, Xn: np.ndarray, ts: np.ndarray, records: list[dict] | None, explain: bool) -> dict:
        """Ядро инференса на numpy: прогноз, калибровка, точное объяснение."""
        if len(Xn) == 0:  # нет прогнозных точек (например, пустой шард) — пустой ответ, не ошибка
            return {"delay_pred": np.empty(0), "q10": np.empty(0), "q90": np.empty(0), "p_late": np.empty(0),
                    "expected_abs_error": np.empty(0), "risk": []}
        raw = self.model.predict(Xn, num_threads=self.n_threads)
        pred = np.where(ts, self.alpha * raw, raw)
        out = {"delay_pred": pred, **self._calibrate(pred)}
        out["risk"] = [risk_level(d, p) for d, p in zip(pred, out["p_late"])]
        if not explain:
            return out
        contrib = self.model.predict(Xn, pred_contrib=True, num_threads=self.n_threads)
        phi, base = contrib[:, :-1], contrib[:, -1]
        G = {g: phi[:, ii].sum(axis=1) for g, ii in self._gidx.items()}
        rule = pred - raw  # вклад правила «старт рейса»
        out["base_value"], out["rule_contrib"] = base, rule
        for g, v in G.items():
            out[f"contrib_{g}"] = v
        causes, texts, tops = [], [], []
        for i in range(len(pred)):
            gi = {g: float(G[g][i]) for g in G}
            if ts[i]:
                gi["trip_start_rule"] = float(rule[i])
            code = max(gi, key=gi.get) if pred[i] >= base[i] else min(gi, key=gi.get)
            causes.append(code)
            tops.append(self._top_features(Xn[i], phi[i]))
            texts.append(self._text(pred[i], out["q10"][i], out["q90"][i], out["p_late"][i], base[i], gi, ts[i]))
        out["cause_code"] = causes
        out["cause"] = [self._title(c) for c in causes]
        out["explanation"] = texts
        out["top_features"] = tops
        out["patterns"] = [behaviour_patterns(r) for r in records] if records is not None else [[] for _ in causes]
        recs = records if records is not None else [{} for _ in causes]
        out["recommendations"] = [recommendations(r, float(d), c, float(pl)) for r, d, c, pl in zip(recs, pred, causes, out["p_late"])]
        return out

    def predict_frame(self, X: pd.DataFrame, seq: np.ndarray | None = None, explain: bool = True) -> pd.DataFrame:
        """Прогноз, калибровка и (опционально) объяснение для таблицы признаков."""
        Xn = X.reindex(columns=self.feats).to_numpy(dtype=np.float64)
        ts = X["tgt_is_trip_start"].to_numpy() == 1 if "tgt_is_trip_start" in X else np.zeros(len(X), bool)
        records = X.to_dict("records") if explain else None
        return pd.DataFrame(self._core(Xn, ts, records, explain))

    def predict_features(self, f: dict, explain: bool = True) -> dict:
        """Прогноз по словарю признаков одной точки — горячий путь без pandas."""
        Xn = np.array([[f.get(c, np.nan) for c in self.feats]], dtype=np.float64)
        ts = np.array([f.get("tgt_is_trip_start") == 1])
        out = self._core(Xn, ts, [f], explain)
        return {k: (v[0] if isinstance(v, (list, np.ndarray)) else v) for k, v in out.items()}

    def _title(self, code: str) -> str:
        if code == "trip_start_rule":
            return "Начало рейса: выпуск по расписанию"
        return self.groups[code]["title"]

    def _top_features(self, x: np.ndarray, phi: np.ndarray, k: int = 4) -> list[dict]:
        order = np.argsort(-np.abs(phi))[:k]
        return [
            {
                "feature": self.feats[j],
                "title": self.info[self.feats[j]]["title"],
                "value": None if not np.isfinite(x[j]) else round(float(x[j]), 3),
                "value_text": fmt_value(x[j], self.info[self.feats[j]]["unit"]),
                "unit": self.info[self.feats[j]]["unit"],
                "contrib_s": round(float(phi[j]), 1),
            }
            for j in order
        ]

    def _text(self, pred: float, q10: float, q90: float, p_late: float, base: float, gi: dict, trip_start: bool) -> str:
        kind = "опоздание" if pred >= 0 else "опережение"
        head = (
            f"Прогноз: {kind} {fmt_signed(pred)} (80%-интервал {fmt_signed(q10)}…{fmt_signed(q90)}, "
            f"P(опоздание > 2 мин) = {p_late:.0%})."
        )
        items = sorted(((g, v) for g, v in gi.items() if abs(v) >= 5), key=lambda kv: -abs(kv[1]))[:3]
        why = "; ".join(f"{_lower_first(self._title(g))} {fmt_signed(v)}" for g, v in items)
        tail = f" Обычно: {fmt_signed(base)}; вклад факторов: {why}." if why else ""
        if trip_start:
            tail += " Цель — первая остановка рейса: ТС выпускается по расписанию."
        return head + tail

    def predict_point(self, ctx, tr_id: int, T: float, target_stop_id: int, target_plan: float, cur_dev_s: float) -> dict:
        """Онлайн-прогноз одной точки (признаки считаются тем же кодом, что офлайн)."""
        from .features import point_features

        f = point_features(ctx, tr_id, T, target_stop_id, target_plan, cur_dev_s)
        row = self.predict_features(f)
        row["features"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items() if k in self.feats}
        return row


def load_predictor(kind: str | None = None):
    """Продовая модель по умолчанию (``ML_MODEL=compact``), ансамбль — ``ML_MODEL=ensemble``."""
    kind = kind or os.environ.get("ML_MODEL", "compact")
    return DelayPredictor.load() if kind == "ensemble" else CompactPredictor.load()
