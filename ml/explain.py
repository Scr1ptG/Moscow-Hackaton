"""Объяснение прогноза: причины/паттерны сбоя для карточки инцидента.

Вклады признаков считаются LightGBM (``pred_contrib=True`` — точные TreeSHAP
значения) и суммируются по смысловым группам. Группа с наибольшим
положительным вкладом — «предполагаемая причина» опоздания, с наибольшим
отрицательным — причина опережения.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Смысловые группы признаков -> (код, текст для диспетчера).
FACTOR_GROUPS: dict[str, tuple[str, list[str]]] = {
    "accumulated_delay": (
        "Накопленное отклонение от графика",
        ["cur_dev_s", "cur_dev_is0", "det_delay", "detw_delay", "det_delay_mean3", "detw_delay_mean3", "pos_delay", "pos_u",
         "det_lb", "detw_lb", "det_at_last_plan", "cur_minus_det_last", "det_age", "detw_age"],
    ),
    "trend": (
        "ТС продолжает терять/нагонять время на маршруте",
        ["det_slope", "detw_slope", "det_extrap", "detw_extrap", "plan_rate_extrap", "det_plan_rate", "phys_delay", "det_speed_route"],
    ),
    "slow_traffic": (
        "Низкая скорость / простой (затор, длительная посадка)",
        ["speed_last", "speed_mean_2m", "speed_mean_5m", "speed_mean_10m", "stop_share_2m", "stop_share_5m", "stop_share_10m",
         "moved_5m", "moved_10m"],
    ),
    "terminal_layover": (
        "Отстой на конечной / начало нового рейса",
        ["layover_path_s", "tgt_after_layover_s", "layover_end_minus_T", "dist_to_layover_stop", "at_terminal_s", "layover_slack_s",
         "n_trip_starts_path", "max_gap_path", "tgt_is_trip_start", "last_is_trip_end", "tgt_gap_prev"],
    ),
    "segment_history": (
        "Этот участок в прошлых рейсах сегодня",
        ["hist_tgt_delay", "hist_tgt_age", "hist_tgt_delay_mean3", "hist_gain", "hist_gain_mean", "det_plus_hist_gain",
         "hist_tripstart_delay", "hist_tgt_det_share", "hist_tgt_dmin"],
    ),
    "tight_schedule": (
        "Плотный плановый график на участке",
        ["plan_path_s", "plan_path_m", "plan_path_speed", "n_stops_plan", "since_last_plan", "lead_s", "tgt_dist_prev",
         "tgt_pos_in_trip", "tgt_to_trip_end", "trip_len", "det_stops_to_tgt", "detw_stops_to_tgt", "det_plan_to_tgt",
         "detw_plan_to_tgt", "det_next_plan_minus_T", "detw_next_plan_minus_T", "dist_to_tgt", "dist_to_last_plan",
         "dist_to_next_plan", "det_same_trip", "detw_same_trip", "det_n15m", "detw_n15m"],
    ),
    "telematics_loss": (
        "Потеря/деградация телематики (GPS, связь)",
        ["has_telemetry", "age_last_pkt", "age_last_valid", "n_pkt_5m", "n_pkt_15m", "n_pkt_30m", "valid_share_5m",
         "valid_share_15m", "valid_share_30m", "det_rate_recent", "n_plan_recent"],
    ),
    "time_of_day": ("Время суток (час пик)", ["hour"]),
}


def group_contributions(contrib: np.ndarray, feats: list[str]) -> pd.DataFrame:
    """Суммирует SHAP-вклады ``[N, F(+1)]`` по группам :data:`FACTOR_GROUPS`."""
    contrib = np.asarray(contrib)
    if contrib.shape[1] == len(feats) + 1:  # последний столбец — bias
        contrib = contrib[:, :-1]
    idx = {f: i for i, f in enumerate(feats)}
    out = {}
    used = set()
    for code, (_title, cols) in FACTOR_GROUPS.items():
        ii = [idx[c] for c in cols if c in idx]
        used.update(ii)
        out[code] = contrib[:, ii].sum(axis=1) if ii else np.zeros(len(contrib))
    rest = [i for i in range(len(feats)) if i not in used]
    out["other"] = contrib[:, rest].sum(axis=1) if rest else np.zeros(len(contrib))
    return pd.DataFrame(out)


def top_cause(groups_row: pd.Series, predicted_delay: float) -> tuple[str, str]:
    """Главная причина: самый большой вклад в сторону прогноза (опоздание/опережение)."""
    g = groups_row.drop(labels=["other"], errors="ignore")
    code = g.idxmax() if predicted_delay >= 0 else g.idxmin()
    return code, FACTOR_GROUPS.get(code, ("Прочее", []))[0]


def behaviour_patterns(f: dict) -> list[str]:
    """Правила-паттерны поведения ТС, предшествующие сбою (для карточки инцидента)."""
    p = []
    if f.get("stop_share_5m", 0) >= 0.6 and f.get("speed_mean_5m", 99) < 8:
        p.append("Длительный простой за последние 5 мин (затор или долгая посадка)")
    if f.get("speed_mean_2m", 99) < 10 and f.get("speed_mean_10m", 0) > 18:
        p.append("Резкое снижение скорости на подходе (перекрёсток/затор)")
    if (f.get("det_slope") or 0) > 0.15:
        p.append("Отставание растёт: ТС теряет время от остановки к остановке")
    if (f.get("det_slope") or 0) < -0.15:
        p.append("ТС нагоняет график")
    if f.get("n_trip_starts_path", 0) > 0 and (f.get("layover_slack_s") or 1e9) < 120:
        p.append("Недостаточный запас на отстой: следующий рейс начнётся с опозданием")
    if (f.get("valid_share_15m") or 1) < 0.5 or (f.get("age_last_valid") or 0) > 300:
        p.append("Потеря GPS/связи: прогноз по последнему известному состоянию")
    if (f.get("hist_gain_mean") or 0) > 90:
        p.append("На этом участке ТС сегодня стабильно теряет время")
    return p


def recommendations(f: dict, delay: float, cause_code: str | None, p_late: float | None = None) -> list[str]:
    """Рекомендации диспетчеру по прогнозу и его причине (правила, 0–3 пункта).

    Уровни согласованы со светофором риска: «критично» (опоздание >= 2 мин или
    P >= 60%) — активные меры; «внимание» (>= 1 мин или P >= 30%) — наблюдение и
    профилактика; опережение — удержание для выравнивания интервала.
    """
    def num(k):
        v = f.get(k)
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    p = p_late or 0.0
    recs: list[str] = []
    slack = num("layover_slack_s")
    if delay >= 120 or p >= 0.6:
        if slack is not None and slack < 300:
            recs.append("Запаса на отстой не хватит: подготовить резервное ТС на следующий рейс или сократить отстой на конечной")
        if cause_code == "traffic" or (num("stop_share_5m") or 0) >= 0.6:
            recs.append("Проверить обстановку на участке (затор, ДТП): рассмотреть объезд или приоритет на светофорах")
        if cause_code == "segment_history":
            recs.append("Участок сегодня систематически «съедает» время: пересмотреть норматив времени хода в расписании")
        if cause_code in ("current_delay", "position") or not recs:
            recs.append("Предупредить водителя: сократить стоянки без посадки; при росте отставания — выпуск резервного ТС")
        recs.append(f"Информировать пассажиров на целевой остановке: ожидаемое опоздание ~{max(1, round(delay / 60))} мин")
    elif delay >= 60 or p >= 0.3:
        recs.append(f"Наблюдать: вероятность опоздания более 2 мин — {round(p * 100)}%; при сохранении тренда предупредить водителя")
        if cause_code == "traffic":
            recs.append("Проверить обстановку на участке: признаки затора/долгой посадки")
        elif cause_code == "segment_history":
            recs.append("На этом участке сегодня уже теряли время — заложить запас на следующих рейсах")
    elif delay <= -90:
        recs.append(f"ТС идёт с опережением ~{round(-delay / 60)} мин: удержать на остановке для выравнивания интервала")
    if cause_code == "telematics" or (num("valid_share_30m") is not None and num("valid_share_30m") < 0.5):
        recs.append("Нет достоверного GPS: связаться с водителем, проверить бортовой терминал")
    return recs[:3]
