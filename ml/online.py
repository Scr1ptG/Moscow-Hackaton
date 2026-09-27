"""Онлайн-состояние для потоковой телеметрии (NDTP -> прогноз в реальном времени).

:class:`OnlineState` накапливает навигационные записи по ТС и по запросу
собирает тот же :class:`ml.features.Context`, что и офлайн-пайплайн, —
поэтому признаки при обучении и в проде считаются одним кодом (нет
train/serving skew; проверяется ``tests/test_no_leak.py``).

Производительность:

* трек ТС хранится numpy-массивами и растёт инкрементально (без пересортировки
  всей истории на каждом запросе), срез ``t <= now`` — бинарным поиском;
* детекция прибытий кэшируется: окно остановки ``[plan-5мин, plan+14мин]``
  после закрытия больше не меняется, пересчитываются только «открытые» окна;
* «широкая» детекция (радиус 30 м) нужна только исследовательскому ансамблю —
  для компактной модели отключается (``wide=False``).

Деградация при обрыве связи: если пакеты от ТС перестали приходить, признаки
``age_last_pkt`` / ``valid_share_*`` это отражают, и модель (обученная в т.ч.
на периодах без GPS) работает по последнему известному состоянию и плану.
"""
from __future__ import annotations

import threading
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .features import R_ARR, R_ARR_WIDE, Context
from .geo import to_xy
from .schedule import prepare_schedule
from .telemetry import Track, first_entries

WINDOW_AFTER_S = 14 * 60
WINDOW_BEFORE_S = 5 * 60
ARR_COLS = ("arr", "arr_known", "arr_w", "arr_w_known", "dmin")


def shard_of(tr_id: int, n: int) -> int:
    """Номер шарда ТС: CRC32 от id (равномерно и детерминированно — одинаково в ingest и ML-сервисе)."""
    return zlib.crc32(str(int(tr_id)).encode()) % n


class _VehicleTrack:
    """Инкрементальный трек одного ТС (numpy-массивы, отсортированы по времени)."""

    def __init__(self):
        self.pending: list[tuple] = []
        self.a = np.empty((0, 6))  # t, valid, lon, lat, speed, heading
        self.xy = np.empty((0, 2))

    def flush(self):
        if not self.pending:
            return
        new = np.array(self.pending, dtype="float64")
        self.pending = []
        if len(self.a) and new[:, 0].min() <= self.a[-1, 0]:  # пришло не по порядку/дубль — пересортировка
            a = np.concatenate([self.a, new])
            a = a[np.argsort(a[:, 0], kind="stable")]
        else:
            a = np.concatenate([self.a, new[np.argsort(new[:, 0], kind="stable")]])
        # дубли по времени: оставляем последний пакет (как офлайн-загрузчик ml.data.load_traffic)
        keep = np.r_[a[1:, 0] != a[:-1, 0], True] if len(a) else np.ones(0, bool)
        self.a = a[keep]
        a = self.a
        x, y = to_xy(a[:, 2], a[:, 3])
        self.xy = np.column_stack([x, y])

    def track(self, tr_id: int, now: float) -> Track | None:
        self.flush()
        n = int(np.searchsorted(self.a[:, 0], now, side="right"))
        if n == 0:
            return None
        a, xy = self.a[:n], self.xy[:n]
        v = a[:, 1].astype(bool)
        return Track(tr_id, a[:, 0], v, a[:, 4], a[v, 0], xy[v, 0], xy[v, 1], a[v, 2], a[v, 3], a[v, 4], a[v, 5])


class OnlineState:
    """Потокобезопасное хранилище телеметрии и плана для онлайн-инференса."""

    def __init__(self, schedule_plan: pd.DataFrame, unit_to_tr: dict[int, int] | None = None, wide: bool = True,
                 shard: tuple[int, int] | None = None):
        """:param shard: ``(i, n)`` — реплика хранит только ТС с ``shard_of(tr_id, n) == i`` (горизонтальное масштабирование)."""
        self.shard = shard
        if shard is not None:
            i, n = shard
            schedule_plan = schedule_plan[schedule_plan["tr_id"].map(lambda t: shard_of(t, n)) == i]
        self.schedule = prepare_schedule(schedule_plan)
        self.unit_to_tr = unit_to_tr or {}
        self.wide = wide
        self._tracks: dict[int, _VehicleTrack] = defaultdict(_VehicleTrack)
        # кэш прибытий: по ТС массивы ARR_COLS + маска «окно закрыто, значение окончательное»
        self._arr = {tr: {c: np.full(len(st), np.nan) for c in ARR_COLS} for tr, st in self.schedule.items()}
        self._final = {tr: np.zeros(len(st), bool) for tr, st in self.schedule.items()}
        self._static = {tr: {c: st[c].to_numpy() for c in st.columns} for tr, st in self.schedule.items()}
        self._lock = threading.Lock()
        self.last_time: float = 0.0

    # ------------------------------------------------------------ приём данных
    def add_record(self, tr_id: int | None, t: float, valid: bool, lon: float, lat: float, speed: float, heading: float = np.nan, unit_id: int | None = None):
        """Добавляет навигационную запись (``tr_id`` можно не знать — тогда по ``unit_id``)."""
        if tr_id is None:
            tr_id = self.unit_to_tr.get(int(unit_id)) if unit_id is not None else None
            if tr_id is None:
                return
        if self.shard is not None and shard_of(int(tr_id), self.shard[1]) != self.shard[0]:
            return  # ТС другой реплики
        if not valid or not np.isfinite(lon) or abs(lon) < 1 or abs(lat) < 1:
            valid, lon, lat, heading = False, np.nan, np.nan, np.nan
        if not np.isfinite(speed) or speed > 150:
            speed = np.nan
        with self._lock:
            self._tracks[int(tr_id)].pending.append((float(t), float(valid), lon, lat, speed, heading))
            self.last_time = max(self.last_time, float(t))

    def add_records(self, records: list[dict]):
        for r in records:
            self.add_record(r.get("tr_id"), r["t"], r.get("valid", True), r.get("lon", np.nan), r.get("lat", np.nan), r.get("speed", np.nan), r.get("heading", np.nan), r.get("unit_id"))

    @property
    def vehicles_with_telemetry(self) -> int:
        return len(self._tracks)

    # ------------------------------------------------------------ контекст
    def context(self, now: float, tr_ids: list[int] | None = None) -> Context:
        """Собирает :class:`Context` на момент ``now`` (только данные ``t <= now``)."""
        tracks, sched = {}, {}
        with self._lock:
            ids = tr_ids or [tr for tr in self.schedule if tr in self._tracks]
            for tr in ids:
                if tr not in self.schedule:
                    continue
                trk = self._tracks[tr].track(tr, now) if tr in self._tracks else None
                tracks[tr] = trk
                sched[tr] = self._schedule_frame(tr, trk, now)
        return Context(schedule=sched, tracks=tracks)

    def _schedule_frame(self, tr: int, trk: Track | None, now: float) -> pd.DataFrame:
        s = self._static[tr]
        A, final = self._arr[tr], self._final[tr]
        plan = s["plan"]
        vals = {c: A[c].copy() for c in ARR_COLS}
        open_idx = np.flatnonzero(~final & (plan - WINDOW_BEFORE_S <= now))
        if len(open_idx) and trk is not None:
            x, y, p = s["x"][open_idx], s["y"][open_idx], plan[open_idx]
            arr, known, dm = first_entries(trk, p, x, y, R_ARR)
            if self.wide:
                arr_w, known_w, dm = first_entries(trk, p, x, y, R_ARR_WIDE)
            else:
                arr_w = known_w = np.full(len(open_idx), np.nan)
            for c, v in zip(ARR_COLS, (arr, known, arr_w, known_w, dm)):
                vals[c][open_idx] = v
            # окно закрыто и трек дошёл до его конца — значение окончательное
            done = open_idx[p + WINDOW_AFTER_S < min(now, trk.t_all[-1])]
            for c in ARR_COLS:
                A[c][done] = vals[c][done]
            final[done] = True
        return pd.DataFrame({**s, **vals})

    # ------------------------------------------------------------ прогнозные точки
    def prediction_points(self, now: float, ctx: Context | None = None) -> pd.DataFrame:
        """Прогнозные точки на ``now``: для каждого ТС — первая остановка с планом в (now+10, now+15] мин.

        ``cur_dev_s`` онлайн оценивается по GPS-прибытиям: отклонение на последней
        плановой остановке (если ТС её прошло), иначе на последней детектированной.
        """
        ctx = ctx or self.context(now)
        from .features import arrivals_known_by

        rows = []
        for tr, st in ctx.schedule.items():
            plan = st["plan"].to_numpy()
            m = np.flatnonzero((plan > now + 600) & (plan <= now + 900))
            if not len(m):
                continue
            j = int(m[0])
            i_last = int(np.searchsorted(plan, now, side="right")) - 1
            arr = arrivals_known_by(st, now, "arr")
            cur = 0.0
            if i_last >= 0:
                if arr[i_last] <= now:
                    cur = float(arr[i_last] - plan[i_last])
                else:
                    done = np.flatnonzero(arr[: i_last + 1] <= now)
                    if len(done):
                        cur = float(arr[done[-1]] - plan[done[-1]])
            rows.append(
                dict(tr_id=tr, T=now, target_stop_id=int(st["stop_id"].iat[j]), target_plan=float(plan[j]), cur_dev_s=round(cur))
            )
        return pd.DataFrame(rows)
