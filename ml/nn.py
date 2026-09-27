"""PyTorch-модель: GRU по окну телеметрии + MLP по табличным признакам.

Вход:

* последовательность последних ``SEQ_LEN * STEP_S`` секунд до ``T`` на
  регулярной сетке (по умолчанию 90 шагов по 20 с = 30 мин), каналы
  :data:`SEQ_CHANNELS`;
* табличные признаки из :mod:`ml.features` (робастная нормировка, NaN -> 0 +
  индикаторы пропуска).

Выход — задержка на целевой остановке, лосс L1 (метрика соревнования — MAE).
Модель экспортируется в ONNX для инференса с низкой задержкой.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SplitData
from .features import Context, _causal_arrivals, arrivals_known_by

SEQ_LEN = 90
STEP_S = 20.0
SEQ_CHANNELS = [
    "has_pkt",  # пришёл ли пакет в пределах шага
    "valid",  # валидны ли координаты
    "speed",  # скорость / 50 км/ч
    "dist_tgt",  # расстояние до целевой остановки, км
    "arr_evt",  # детектировано прибытие на остановку на этом шаге
    "arr_delay",  # задержка на этой остановке / 300 с
    "plan_evt",  # плановое время какой-то остановки на этом шаге
    "trip_start_evt",  # плановое начало рейса на этом шаге
]


def point_sequence(ctx: Context, tr_id: int, T: float, target_stop_id: int) -> np.ndarray:
    """Окно телеметрии до ``T`` в виде массива ``[SEQ_LEN, len(SEQ_CHANNELS)]``."""
    seq = np.zeros((SEQ_LEN, len(SEQ_CHANNELS)), dtype=np.float32)
    grid = T - (SEQ_LEN - 1 - np.arange(SEQ_LEN)) * STEP_S
    tr = ctx.tracks.get(tr_id)
    st = ctx.schedule.get(tr_id)
    if st is None:
        return seq
    ids = st["stop_id"].to_numpy()
    pos = np.flatnonzero(ids == target_stop_id)
    j = int(pos[0]) if len(pos) else len(st) - 1
    sx, sy = st["x"].to_numpy()[j], st["y"].to_numpy()[j]
    plan = st["plan"].to_numpy()
    if tr is not None and len(tr.t_all):
        ia = np.searchsorted(tr.t_all, grid, side="right") - 1
        ok = (ia >= 0) & (grid - tr.t_all[np.clip(ia, 0, None)] < STEP_S)
        seq[:, 0] = ok
        seq[:, 1] = np.where(ok, tr.valid_all[np.clip(ia, 0, None)], 0)
        spd = tr.speed_all[np.clip(ia, 0, None)]
        seq[:, 2] = np.where(ok & np.isfinite(spd), np.nan_to_num(spd) / 50.0, 0)
        iv = np.searchsorted(tr.t, grid, side="right") - 1
        okv = (iv >= 0) & (grid - tr.t[np.clip(iv, 0, None)] < 90)
        if len(tr.t):
            d = np.hypot(tr.x[np.clip(iv, 0, None)] - sx, tr.y[np.clip(iv, 0, None)] - sy) / 1000.0
            seq[:, 3] = np.where(okv, np.minimum(d, 20.0), 0)
    lo = max(0, int(np.searchsorted(plan, T - SEQ_LEN * STEP_S - 900)))
    hi = j + 1
    a = _causal_arrivals(arrivals_known_by(st, T, "arr"), T, lo, hi)
    for r, t_arr in enumerate(a):
        if np.isnan(t_arr) or t_arr < grid[0]:
            continue
        g = int(round((t_arr - grid[0]) / STEP_S))
        seq[g, 4] = 1.0
        seq[g, 5] = np.clip((t_arr - plan[lo + r]) / 300.0, -3, 3)
    tstart = st["is_trip_start"].to_numpy()
    m = (plan >= grid[0]) & (plan <= T)
    for p, s in zip(plan[m], tstart[m]):
        g = int(round((p - grid[0]) / STEP_S))
        seq[g, 6] = 1.0
        if s:
            seq[g, 7] = 1.0
    return seq


def build_sequences(split: SplitData, ctx: Context) -> np.ndarray:
    """Последовательности для всех прогнозных точек сплита ``[N, L, C]``."""
    return np.stack(
        [point_sequence(ctx, int(p.tr_id), float(p.T), int(p.target_stop_id)) for p in split.points.itertuples(index=False)]
    )


@dataclass
class TabScaler:
    """Робастная нормировка табличных признаков (медиана / IQR, NaN -> 0)."""

    cols: list[str]
    med: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "TabScaler":
        x = df[cols].to_numpy(dtype="float64")
        med = np.nanmedian(x, axis=0)
        q = np.nanpercentile(x, [25, 75], axis=0)
        iqr = np.where(np.isfinite(q[1] - q[0]) & (q[1] - q[0] > 1e-9), q[1] - q[0], 1.0)
        return cls(cols, np.nan_to_num(med), iqr)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].to_numpy(dtype="float64")
        miss = np.isnan(x)
        z = np.clip((x - self.med) / self.iqr, -5, 5)
        z[miss] = 0.0
        return np.concatenate([z, miss.astype("float64")], axis=1).astype(np.float32)


def make_model(n_tab: int, n_ch: int = len(SEQ_CHANNELS), hidden: int = 64):
    """GRU-энкодер последовательности + MLP по табличным признакам."""
    import torch
    from torch import nn

    class DelayNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_ch, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(hidden + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, seq, tab):
            h, _ = self.gru(torch.relu(self.inp(seq)))
            z = torch.cat([h[:, -1], self.tab(tab)], dim=1)
            return self.head(z).squeeze(1) * 100.0  # масштаб выхода — секунды

    return DelayNet()


def fit_nn(seq, tab, y, w=None, epochs: int = 60, lr: float = 2e-3, batch: int = 128, seed: int = 0, device: str | None = None):
    """Обучает :func:`make_model` с лоссом L1 (AdamW + косинусный LR)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(tab.shape[1], seq.shape[2]).to(device)
    S = torch.tensor(seq, device=device)
    X = torch.tensor(tab, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    W = torch.tensor(w if w is not None else np.ones(len(y)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=device)
        for i in range(0, len(y), batch):
            b = perm[i : i + batch]
            loss = (torch.abs(model(S[b], X[b]) - Y[b]) * W[b]).sum() / W[b].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def export_onnx(model, n_tab: int, path, opset: int = 17) -> None:
    """Экспорт модели в ONNX с динамическим размером батча."""
    import torch

    model = model.to("cpu").eval()
    seq = torch.zeros(1, SEQ_LEN, len(SEQ_CHANNELS))
    tab = torch.zeros(1, n_tab)
    torch.onnx.export(
        model,
        (seq, tab),
        str(path),
        input_names=["seq", "tab"],
        output_names=["delay"],
        dynamic_axes={"seq": {0: "batch"}, "tab": {0: "batch"}, "delay": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def predict_nn(model, seq, tab, batch: int = 1024) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            out.append(model(torch.tensor(seq[i : i + batch], device=device), torch.tensor(tab[i : i + batch], device=device)).cpu().numpy())
    return np.concatenate(out)
