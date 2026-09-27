"""Признаки прогнозной точки ``(tr_id, T)`` строго по данным, доступным на ``T``.

Источники (и только они):

* плановое расписание ТС (известно заранее);
* телеметрия с ``t <= T`` (обрезается в :func:`_slice_track`);
* прибытия на остановки, детектированные по GPS с ``arrival <= T``;
* подсказка ``cur_dev_s`` из условия.

Та же функция :func:`point_features` используется и офлайн (CSV), и онлайн
(поток NDTP в backend): онлайн-контур просто передаёт трек, накопленный к ``T``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SplitData
from .geo import haversine, point_segment
from .schedule import prepare_schedule
from .telemetry import Track, build_tracks, detect_arrivals

#: Радиус «точной» детекции прибытия (калибровка по фактам: медиана ошибки ~4 с).
R_ARR = 15.0
#: Радиус «широкой» детекции (больше покрытие, чуть хуже точность).
R_ARR_WIDE = 30.0
#: Смещение Москвы относительно наивного времени раздачи (UTC), ч.
TZ_SHIFT_H = 3


@dataclass
class Context:
    """Предрасчёт по сплиту: план по ТС, треки и прибытия."""

    schedule: dict[int, pd.DataFrame]
    tracks: dict[int, Track]


def prepare_context(split: SplitData) -> Context:
    """Строит план по ТС, треки и детектирует прибытия (на весь трек).

    Детекция считается на полном треке один раз, но для точки ``T`` дальше
    берутся только прибытия, **подтверждённые** к ``T`` (``arr_known <= T``,
    см. :func:`arrivals_known_by`). Это эквивалентно детекции на треке,
    обрезанном по ``T`` (проверяется в ``tests/test_no_leak.py``).
    """
    sched = prepare_schedule(split.schedule)
    tracks = build_tracks(split.traffic)
    for tr, st in sched.items():
        a = detect_arrivals(tracks.get(tr), st, R=R_ARR, monotone=False)
        w = detect_arrivals(tracks.get(tr), st, R=R_ARR_WIDE, monotone=False)
        st["arr"] = a["arr"].to_numpy()
        st["arr_known"] = a["arr_known"].to_numpy()
        st["arr_w"] = w["arr"].to_numpy()
        st["arr_w_known"] = w["arr_known"].to_numpy()
        st["dmin"] = w["dmin"].to_numpy()
    return Context(schedule=sched, tracks=tracks)


def arrivals_known_by(st: pd.DataFrame, T: float, col: str = "arr") -> np.ndarray:
    """Времена прибытий, о которых на момент ``T`` уже известно (иначе NaN)."""
    known = st[f"{col}_known"].to_numpy()
    return np.where(known <= T, st[col].to_numpy(), np.nan)


def _slice_track(tr: Track | None, T: float):
    """Возвращает срезы трека с ``t <= T`` (все пакеты и валидные точки)."""
    if tr is None:
        return None
    ia = int(np.searchsorted(tr.t_all, T, side="right"))
    iv = int(np.searchsorted(tr.t, T, side="right"))
    return ia, iv


def _causal_arrivals(arr: np.ndarray, T: float, lo: int, hi: int) -> np.ndarray:
    """Прибытия ``<= T`` на остановках ``[lo, hi)`` с причинным фильтром порядка.

    Прибытия идут в порядке остановок; детекция раньше уже принятой предыдущей
    остановки (минус 30 с допуска) считается ложной (например, ТС проехало
    рядом с остановкой следующего рейса).
    """
    a = arr[lo:hi].copy()
    a[~(a <= T)] = np.nan
    run = -np.inf
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        if a[i] + 30 < run:
            a[i] = np.nan
        else:
            run = max(run, a[i])
    return a


def _window_stats(t: np.ndarray, v: np.ndarray, T: float, win: float):
    m = t > T - win
    if not m.any():
        return np.nan, np.nan, np.nan
    x = v[m]
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(x.mean()), float((x < 3).mean()), float(np.median(x))


def point_features(ctx: Context, tr_id: int, T: float, target_stop_id: int, target_plan: float, cur_dev_s: float) -> dict:
    """Считает признаки одной прогнозной точки.

    :returns: словарь «имя признака -> значение» (NaN, если признак не определён).
    """
    f: dict[str, float] = {}
    st = ctx.schedule.get(tr_id)
    tr = ctx.tracks.get(tr_id)

    # ---------- время и подсказка ----------
    f["lead_s"] = target_plan - T
    f["cur_dev_s"] = cur_dev_s
    f["cur_dev_is0"] = float(cur_dev_s == 0)
    f["hour"] = ((T / 3600.0) + TZ_SHIFT_H) % 24

    if st is None:
        return f
    plan = st["plan"].to_numpy()
    ids = st["stop_id"].to_numpy()
    pos = np.flatnonzero(ids == target_stop_id)
    j = int(pos[0]) if len(pos) else int(np.searchsorted(plan, target_plan))
    j = min(j, len(st) - 1)
    i_last = int(np.searchsorted(plan, T, side="right")) - 1  # последняя остановка с планом <= T

    # прибытия, подтверждённые GPS к моменту T (единственный вид прибытий в признаках)
    arr_T = arrivals_known_by(st, T, "arr")
    arr_w_T = arrivals_known_by(st, T, "arr_w")
    gap = st["gap_prev"].to_numpy()
    dist = st["dist_prev"].to_numpy()
    cum = st["cum_dist"].to_numpy()
    trip = st["trip_no"].to_numpy()
    tstart = st["is_trip_start"].to_numpy()
    pos_in_trip = st["pos_in_trip"].to_numpy()
    trip_len = st["trip_len"].to_numpy()

    # ---------- план: путь от «текущей» плановой остановки до целевой ----------
    f["n_stops_plan"] = j - i_last
    f["since_last_plan"] = T - plan[i_last] if i_last >= 0 else np.nan
    seg = slice(max(i_last, 0) + 1, j + 1)
    f["plan_path_s"] = target_plan - plan[i_last] if i_last >= 0 else np.nan
    f["plan_path_m"] = cum[j] - cum[i_last] if i_last >= 0 else np.nan
    f["plan_path_speed"] = f["plan_path_m"] / f["plan_path_s"] if i_last >= 0 and f["plan_path_s"] > 0 else np.nan
    gseg = np.nan_to_num(gap[seg])
    f["max_gap_path"] = float(gseg.max()) if len(gseg) else 0.0
    f["n_trip_starts_path"] = float(tstart[seg].sum())
    f["layover_path_s"] = float(gseg[tstart[seg]].sum()) if len(gseg) else 0.0
    f["tgt_is_trip_start"] = float(tstart[j])
    f["tgt_gap_prev"] = gap[j]
    f["tgt_dist_prev"] = dist[j]
    f["tgt_pos_in_trip"] = float(pos_in_trip[j])
    f["tgt_to_trip_end"] = float(trip_len[j] - 1 - pos_in_trip[j])
    f["trip_len"] = float(trip_len[j])
    f["last_is_trip_end"] = float(i_last >= 0 and i_last + 1 < len(st) and tstart[i_last + 1])
    f["trip_no"] = float(trip[j])
    # время от конца отстоя (начала рейса) до целевой — сколько «ехать» после отстоя
    if f["n_trip_starts_path"] > 0:
        k0 = seg.start + int(np.flatnonzero(tstart[seg])[-1])
        f["tgt_after_layover_s"] = target_plan - plan[k0]
        f["layover_end_minus_T"] = plan[k0] - T
    else:
        f["tgt_after_layover_s"] = np.nan
        f["layover_end_minus_T"] = np.nan

    # ---------- телеметрия <= T ----------
    sl = _slice_track(tr, T)
    if sl is None or sl[0] == 0:
        f["has_telemetry"] = 0.0
        return f
    ia, iv = sl
    f["has_telemetry"] = 1.0
    ta = tr.t_all[:ia]
    va = tr.valid_all[:ia]
    f["age_last_pkt"] = T - ta[-1]
    f["age_last_valid"] = T - tr.t[iv - 1] if iv > 0 else np.nan
    for w in (300, 900, 1800):
        m = ta > T - w
        f[f"n_pkt_{w // 60}m"] = float(m.sum())
        f[f"valid_share_{w // 60}m"] = float(va[m].mean()) if m.any() else np.nan

    if iv > 0:
        tv, xv, yv = tr.t[:iv], tr.x[:iv], tr.y[:iv]
        sp = tr.speed[:iv]
        f["speed_last"] = sp[-1]
        for w in (120, 300, 600):
            mean, stop_share, _ = _window_stats(tv, sp, T, w)
            f[f"speed_mean_{w // 60}m"] = mean
            f[f"stop_share_{w // 60}m"] = stop_share
        for w in (300, 600):
            m = tv > T - w
            if m.sum() >= 2:
                f[f"moved_{w // 60}m"] = float(np.hypot(np.diff(xv[m]), np.diff(yv[m])).sum())
            else:
                f[f"moved_{w // 60}m"] = np.nan
        lx, ly = xv[-1], yv[-1]
        sx, sy = st["x"].to_numpy(), st["y"].to_numpy()
        f["dist_to_tgt"] = float(np.hypot(lx - sx[j], ly - sy[j]))
        if i_last >= 0:
            f["dist_to_last_plan"] = float(np.hypot(lx - sx[i_last], ly - sy[i_last]))
        if i_last + 1 < len(st):
            f["dist_to_next_plan"] = float(np.hypot(lx - sx[i_last + 1], ly - sy[i_last + 1]))

    # ---------- прибытия по GPS (<= T) ----------
    lo = max(0, min(i_last, j) - 40)
    hi = j + 1
    for name, a_src in (("det", arr_T), ("detw", arr_w_T)):
        # старт рейса не годится как опорная остановка: ТС стоит на конечной
        # в зоне остановки задолго до отправления, и «задержка» там не
        # отражает текущее отставание от графика
        a_all = a_src.copy()
        a_all[tstart] = np.nan
        a = _causal_arrivals(a_all, T, lo, hi)
        ok = np.flatnonzero(~np.isnan(a))
        if len(ok) == 0:
            f[f"{name}_delay"] = np.nan
            continue
        k_rel = ok[-1]
        k = lo + k_rel
        d = a[ok] - plan[lo + ok]
        f[f"{name}_delay"] = d[-1]
        f[f"{name}_age"] = T - a[k_rel]
        f[f"{name}_stops_to_tgt"] = float(j - k)
        f[f"{name}_plan_to_tgt"] = target_plan - plan[k]
        f[f"{name}_same_trip"] = float(trip[k] == trip[j])
        f[f"{name}_delay_mean3"] = float(d[-3:].mean())
        recent = a[ok] > T - 900
        f[f"{name}_n15m"] = float(recent.sum())
        if recent.sum() >= 3:
            tt = plan[lo + ok][recent]
            dd = d[recent]
            f[f"{name}_slope"] = float(np.polyfit(tt - tt.mean(), dd, 1)[0]) if np.ptp(tt) > 0 else 0.0
        # нижняя граница задержки: следующую остановку ТС ещё не прошло
        if k + 1 < len(st):
            f[f"{name}_lb"] = max(0.0, T - plan[k + 1])
            f[f"{name}_next_plan_minus_T"] = plan[k + 1] - T
        # экстраполяция тренда задержки до целевой (бустингу трудно учить произведения)
        if not np.isnan(f.get(f"{name}_slope", np.nan)):
            f[f"{name}_extrap"] = float(np.clip(d[-1] + f[f"{name}_slope"] * (target_plan - plan[k]), -900, 900))
        # «физический» ETA: темп продвижения по маршруту за последние ~15 мин
        if name == "det" and trip[k] == trip[j]:
            ok15 = ok[(a[ok] > T - 1200) & (trip[lo + ok] == trip[k])]
            if len(ok15) >= 2 and a[ok15[-1]] - a[ok15[0]] > 120:
                k_a, k_b = lo + ok15[0], lo + ok15[-1]
                v = (cum[k_b] - cum[k_a]) / (a[ok15[-1]] - a[ok15[0]])
                f["det_speed_route"] = v
                f["det_plan_rate"] = (plan[k_b] - plan[k_a]) / (a[ok15[-1]] - a[ok15[0]])
                if v > 0.5:
                    eta = a[ok15[-1]] + (cum[j] - cum[k_b]) / v
                    f["phys_delay"] = float(np.clip(eta - target_plan, -900, 1500))
                f["plan_rate_extrap"] = float(
                    np.clip(d[-1] + (target_plan - plan[k]) * (1.0 / max(f["det_plan_rate"], 0.2) - 1.0), -900, 1500)
                )

    # доля недавних плановых остановок, где GPS «увидел» ТС: прокси того, что АСДУ
    # регистрирует прибытия автоматически (иначе факт заполняют вручную: 0 / копия)
    recent_idx = np.flatnonzero((plan > T - 1500) & (plan < T - 240))
    if len(recent_idx):
        aw = arr_w_T[recent_idx]
        f["det_rate_recent"] = float(np.mean(aw <= T))
        f["n_plan_recent"] = float(len(recent_idx))

    # сверка подсказки с GPS: если ТС уже прошло «плановую» остановку, то факт там
    # должен совпасть с нашей детекцией; расхождение — признак ручного заполнения
    if i_last >= 0:
        a_last = arr_T[i_last]
        if a_last <= T:
            f["det_at_last_plan"] = a_last - plan[i_last]
            f["cur_minus_det_last"] = cur_dev_s - f["det_at_last_plan"]

    # отстой на пути к целевой: где ТС относительно конечной и хватит ли запаса
    if f["n_trip_starts_path"] > 0:
        k0 = seg.start + int(np.flatnonzero(tstart[seg])[-1])
        sx, sy = st["x"].to_numpy(), st["y"].to_numpy()
        if iv > 0:
            dterm = float(np.hypot(tr.x[iv - 1] - sx[k0], tr.y[iv - 1] - sy[k0]))
            f["dist_to_layover_stop"] = dterm
            # сколько последних секунд ТС находится у конечной (< 200 м)
            m = np.hypot(tr.x[:iv] - sx[k0], tr.y[:iv] - sy[k0]) < 200
            if m[-1]:
                first_out = np.flatnonzero(~m)
                t_in = tr.t[first_out[-1] + 1] if len(first_out) else tr.t[0]
                f["at_terminal_s"] = T - t_in
            else:
                f["at_terminal_s"] = 0.0
        run = f.get("det_delay", np.nan)
        if np.isnan(run):
            run = cur_dev_s
        # ожидаемый запас: план отправления минус ожидаемое прибытие на конечную
        f["layover_slack_s"] = plan[k0] - (plan[k0 - 1] + run) if k0 > 0 else np.nan

    # задержка «по положению»: проекция последней точки на отрезок k -> k+1
    if iv > 0 and not np.isnan(f.get("det_delay", np.nan)):
        k = j - int(f["det_stops_to_tgt"])
        if k + 1 < len(st) and trip[k + 1] == trip[k]:
            sx, sy = st["x"].to_numpy(), st["y"].to_numpy()
            dseg, u = point_segment(tr.x[iv - 1], tr.y[iv - 1], sx[k], sy[k], sx[k + 1], sy[k + 1])
            if dseg < 150:
                p_at = plan[k] + float(u) * (plan[k + 1] - plan[k])
                f["pos_delay"] = tr.t[iv - 1] - p_at
                f["pos_u"] = float(u)

    # ---------- история того же дня: та же остановка в прошлых рейсах ----------
    lk = st["loc_key"].to_numpy()
    arr = arr_T
    # прошлые визиты той же остановки (то же место и та же позиция в рейсе),
    # план которых уже в прошлом: детектировали ли мы там ТС (иначе АСДУ, вероятно,
    # тоже не видит ТС и факт будет заполнен вручную)
    prev_vis = np.flatnonzero((lk == lk[j]) & (pos_in_trip == pos_in_trip[j]) & (plan < min(plan[j] - 1200, T - 900)))
    if len(prev_vis):
        pv = prev_vis[-4:]
        f["hist_tgt_det_share"] = float(np.mean(arr[pv] <= T))
        dm = st["dmin"].to_numpy()[pv]
        f["hist_tgt_dmin"] = float(np.nanmedian(dm)) if np.isfinite(dm).any() else np.nan
    same = np.flatnonzero((lk == lk[j]) & (pos_in_trip == pos_in_trip[j]) & (plan < plan[j] - 1200) & (arr <= T))
    if len(same):
        jj = same[-1]
        f["hist_tgt_delay"] = arr[jj] - plan[jj]
        f["hist_tgt_age"] = T - arr[jj]
        f["hist_tgt_delay_mean3"] = float(np.mean(arr[same[-3:]] - plan[same[-3:]]))
        k = j - int(f.get("det_stops_to_tgt", np.nan)) if not np.isnan(f.get("det_stops_to_tgt", np.nan)) else None
        if k is not None:
            gains = []
            for jj in same[::-1][:3]:
                kk = jj - (j - k)
                if 0 <= kk < len(st) and lk[kk] == lk[k] and arr[kk] <= T:
                    gains.append((arr[jj] - plan[jj]) - (arr[kk] - plan[kk]))
            if gains:
                f["hist_gain"] = gains[0]
                f["hist_gain_mean"] = float(np.mean(gains))
                if not np.isnan(f.get("det_delay", np.nan)):
                    f["det_plus_hist_gain"] = f["det_delay"] + float(np.mean(gains))
    trip_starts_done = np.flatnonzero(tstart & (arr <= T) & (plan < T))
    if len(trip_starts_done):
        f["hist_tripstart_delay"] = float(np.mean(arr[trip_starts_done[-2:]] - plan[trip_starts_done[-2:]]))
    return f


def build_features(split: SplitData, ctx: Context | None = None) -> pd.DataFrame:
    """Признаки для всех прогнозных точек сплита (+ служебные колонки)."""
    ctx = ctx or prepare_context(split)
    rows = []
    for p in split.points.itertuples(index=False):
        rows.append(point_features(ctx, int(p.tr_id), float(p.T), int(p.target_stop_id), float(p.target_plan), float(p.cur_dev_s)))
    X = pd.DataFrame(rows)
    meta = split.points.reset_index(drop=True)
    return pd.concat([meta, X.drop(columns=["cur_dev_s"])], axis=1)
