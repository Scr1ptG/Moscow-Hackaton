"""Оценка объяснимости компактной модели и графики для документации/жюри.

Проверки (результат — ``reports/explainability.json``, графики — ``reports/figures``):

1. **Точность (аддитивность)**: база + сумма вкладов + правило = прогноз.
2. **Монотонность**: для признаков с ограничением прогноз не убывает (не растёт)
   при движении значения по сетке, остальные признаки фиксированы.
3. **Верность (deletion test)**: если «нейтрализовать» главную по объяснению группу
   (заменить её признаки медианой), прогноз меняется сильнее, чем при
   нейтрализации случайной другой группы.
4. **Устойчивость**: у соседних по времени точек одного ТС (шаг 5 мин) главная
   причина и вектор вкладов похожи.
5. **Калибровка вне выборки**: P(опоздание) и интервал q10–q90, таблица калибровки
   строится без ТС, на котором проверяется (leave-one-vehicle-out).

Запуск: ``python -m ml.explain_eval``
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .compact import FEATS, FEATURES, REPORT_DIR, calibrated, calibration_table, load_labeled
from .predictor import CompactPredictor

FIG_DIR = REPORT_DIR / "figures"

# эталонная палитра (dataviz reference palette, светлая тема)
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"


def _group_matrix(out: pd.DataFrame, pr: CompactPredictor) -> pd.DataFrame:
    return out[[f"contrib_{g}" for g in pr.groups]].rename(columns=lambda c: c.replace("contrib_", ""))


def check_additivity(out: pd.DataFrame, pr: CompactPredictor) -> float:
    G = _group_matrix(out, pr).sum(axis=1)
    return float(np.max(np.abs(out["base_value"] + G + out["rule_contrib"] - out["delay_pred"])))


def check_monotonic(X: pd.DataFrame, pr: CompactPredictor, n: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = X.iloc[rng.choice(len(X), min(n, len(X)), replace=False)].reindex(columns=pr.feats).to_numpy(float)
    res = {}
    for j, f in enumerate(pr.feats):
        mono = FEATURES[f][2]
        if mono == 0:
            continue
        v = X[f].dropna()
        grid = np.quantile(v, np.linspace(0.01, 0.99, 25))
        viol = 0
        for r in rows:
            Z = np.repeat(r[None], len(grid), axis=0)
            Z[:, j] = grid
            p = pr.model.predict(Z, num_threads=1)
            d = np.diff(p) * mono
            viol += int(np.sum(d < -1e-9))
        res[f] = {"direction": mono, "violations": viol, "checks": int(len(rows) * (len(grid) - 1))}
    return res


def check_deletion(X: pd.DataFrame, out: pd.DataFrame, pr: CompactPredictor, seed: int = 0) -> dict:
    """Нейтрализация главной группы vs случайной другой (медианы признаков)."""
    rng = np.random.default_rng(seed)
    med = X[pr.feats].median()
    Xn = X.reindex(columns=pr.feats).to_numpy(float)
    G = _group_matrix(out, pr)
    base_pred = pr.model.predict(Xn, num_threads=1)
    idx = {f: i for i, f in enumerate(pr.feats)}
    d_top, d_rand = [], []
    for i in range(len(X)):
        g = G.iloc[i].abs()
        top = g.idxmax()
        other = rng.choice([k for k in g.index if k != top])
        for grp, acc in ((top, d_top), (other, d_rand)):
            z = Xn[i].copy()
            for f in pr.groups[grp]["features"]:
                z[idx[f]] = med[f]
            acc.append(abs(pr.model.predict(z[None], num_threads=1)[0] - base_pred[i]))
    d_top, d_rand = np.array(d_top), np.array(d_rand)
    return {
        "mean_abs_change_top_group_s": float(d_top.mean()),
        "mean_abs_change_random_group_s": float(d_rand.mean()),
        "share_top_bigger": float(np.mean(d_top > d_rand)),
    }


def check_stability(lab: pd.DataFrame, out: pd.DataFrame, pr: CompactPredictor) -> dict:
    """Соседние точки одного ТС через 5 мин: та же главная причина? похожи ли вклады?"""
    G = _group_matrix(out, pr).to_numpy()
    d = lab[["tr_id", "T"]].reset_index(drop=True).assign(i=np.arange(len(lab)), cause=out["cause_code"].to_numpy())
    key = {(r.tr_id, r.T): r.i for r in d.itertuples()}
    same, cos = [], []
    for r in d.itertuples():
        j = key.get((r.tr_id, r.T + 300))
        if j is None:
            continue
        same.append(d.cause.iat[r.i] == d.cause.iat[j])
        a, b = G[r.i], G[j]
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return {"pairs": len(same), "same_top_cause": float(np.mean(same)), "median_cosine_contrib": float(np.median(cos))}


def check_calibration_oos() -> tuple[dict, pd.DataFrame]:
    """Калибровка вне выборки: таблица строится без проверяемого ТС."""
    oof = pd.read_csv(REPORT_DIR / "cv_oof_compact.csv")
    parts = []
    for tr in oof["tr_id"].unique():
        te = oof["tr_id"] == tr
        calib = calibration_table(oof.loc[~te, "oof"].to_numpy(), oof.loc[~te, "y"].to_numpy())
        c = calibrated(oof.loc[te, "oof"].to_numpy(), calib)
        parts.append(oof[te].assign(**c))
    d = pd.concat(parts)
    late = (d["y"] > 120).astype(float)
    bins = np.linspace(0, 1, 6)
    d["p_bin"] = pd.cut(d["p_late"], bins, include_lowest=True)
    rel = d.groupby("p_bin", observed=True).agg(p_mean=("p_late", "mean"), freq=("y", lambda s: float(np.mean(s > 120))), n=("y", "size")).reset_index()
    res = {
        "interval_q10_q90_coverage": float(np.mean((d["y"] >= d["q10"]) & (d["y"] <= d["q90"]))),
        "brier_p_late": float(np.mean((d["p_late"] - late) ** 2)),
        "brier_constant": float(np.mean((late.mean() - late) ** 2)),
        "expected_vs_actual_abs_error": [float(d["expected_abs_error"].mean()), float(np.mean(np.abs(d["y"] - d["oof"])))],
    }
    return res, rel


# ------------------------------------------------------------------ графики


def _style(ax, title: str, subtitle: str = ""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)


def fig_importance(imp: pd.Series, path):
    import matplotlib.pyplot as plt

    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.barh(range(len(imp)), imp.values, height=0.45, color=BLUE)
    ax.set_yticks(range(len(imp)), imp.index, color=INK, fontsize=9)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.0f} с", va="center", fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("средний |вклад| в прогноз, с", color=INK2, fontsize=9)
    ax.set_xlim(0, imp.max() * 1.15)
    _style(ax, "Что сильнее всего влияет на прогноз задержки", "Средний |SHAP| группы признаков, реальные ТС")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_dependence(X: pd.DataFrame, phi: pd.DataFrame, feats: list[str], path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE)
    for ax, f in zip(axes.ravel(), feats):
        x, y = X[f].to_numpy(float), phi[f].to_numpy(float)
        m = np.isfinite(x)
        lo, hi = np.nanpercentile(x, [1, 99])
        m &= (x >= lo) & (x <= hi)
        ax.axhline(0, color=BASE, linewidth=1)
        ax.scatter(x[m], y[m], s=9, color=BLUE, alpha=0.35, linewidths=0)
        title, unit, mono = FEATURES[f]
        mono_txt = {1: "монотонно ↑", -1: "монотонно ↓", 0: "без ограничения"}[mono]
        _style(ax, _wrap(title, 38), f"{unit}; {mono_txt}")
        ax.title.set_fontsize(9.5)
        ax.set_ylabel("вклад, с", color=INK2, fontsize=8.5)
    fig.suptitle("Как значение признака меняет прогноз (SHAP dependence)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_reliability(rel: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=BASE, linewidth=1)
    ax.plot(rel["p_mean"], rel["freq"], color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for r in rel.itertuples():
        ax.text(r.p_mean + 0.02, r.freq - 0.045, f"n={r.n}", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("предсказанная P(опоздание > 2 мин)", color=INK2, fontsize=9)
    ax.set_ylabel("наблюдаемая доля опозданий", color=INK2, fontsize=9)
    _style(ax, "Калибровка вероятности опоздания", "Вне выборки (leave-one-vehicle-out); диагональ — идеал")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_waterfall(row: pd.Series, pr: CompactPredictor, sample_id: str, path):
    import matplotlib.pyplot as plt

    items = [(pr.groups[g]["title"], row[f"contrib_{g}"]) for g in pr.groups]
    if abs(row["rule_contrib"]) > 0.5:
        items.append(("Правило: старт рейса", row["rule_contrib"]))
    items = [it for it in items if abs(it[1]) >= 1]
    items.sort(key=lambda kv: -abs(kv[1]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(items) + 1.9), facecolor=SURFACE)
    cur = row["base_value"]
    labels = ["Обычное отклонение (база)"]
    ax.barh(0, cur, height=0.45, color=MUTED)
    ax.text(cur, 0, f"  {cur:+.0f} с", va="center", fontsize=8.5, color=INK2)
    for k, (name, v) in enumerate(items, start=1):
        ax.barh(k, v, left=cur, height=0.45, color=RED if v > 0 else BLUE)
        ax.text(max(cur, cur + v), k, f"  {v:+.0f} с", va="center", fontsize=8.5, color=INK2)
        cur += v
        labels.append(name)
    k = len(items) + 1
    ax.barh(k, cur, height=0.45, color=INK)
    ax.text(cur, k, f"  {cur:+.0f} с", va="center", fontsize=8.5, color=INK)
    labels.append("Прогноз")
    ax.axvline(0, color=BASE, linewidth=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.margins(x=0.14)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("отклонение от графика, с (красное — к опозданию, синее — к опережению)", color=INK2, fontsize=8.5)
    _style(ax, f"Разбор прогноза {sample_id}", "Точное разложение: база + вклады факторов = прогноз")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _wrap(s: str, n: int) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return "\n".join(lines + [cur])


def main():
    import matplotlib

    matplotlib.use("Agg")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lab, val = load_labeled()
    real = lab[~lab["is_synthetic"]].reset_index(drop=True)
    pr = CompactPredictor.load()
    out = pr.predict_frame(real)
    Xn = real.reindex(columns=pr.feats).to_numpy(float)
    phi = pd.DataFrame(pr.model.predict(Xn, pred_contrib=True, num_threads=1)[:, :-1], columns=pr.feats)
    G = _group_matrix(out, pr)
    imp_feat = phi.abs().mean().sort_values(ascending=False)
    imp_group = G.abs().mean().sort_values(ascending=False)
    cal, rel = check_calibration_oos()
    res = {
        "additivity_max_abs_error_s": check_additivity(out, pr),
        "monotonicity": check_monotonic(real, pr),
        "deletion_test": check_deletion(real, out, pr),
        "stability_5min": check_stability(real, out, pr),
        "calibration_out_of_sample": cal,
        "importance_feature_mean_abs_s": imp_feat.round(2).to_dict(),
        "importance_group_mean_abs_s": imp_group.round(2).to_dict(),
        "top_cause_share": out["cause_code"].value_counts(normalize=True).round(3).to_dict(),
    }
    (REPORT_DIR / "explainability.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    fig_importance(imp_group.rename(index=lambda g: pr.groups[g]["title"]), FIG_DIR / "importance_groups.png")
    fig_dependence(real, phi, list(imp_feat.index[:6]), FIG_DIR / "shap_dependence.png")
    fig_reliability(rel, FIG_DIR / "calibration_p_late.png")
    i = int(np.argmax(out["delay_pred"].to_numpy() * (real["n_trip_starts_path"].to_numpy() == 0)))
    fig_waterfall(out.iloc[i], pr, real["sample_id"].iat[i], FIG_DIR / "waterfall_example.png")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("importance")}, ensure_ascii=False, indent=2))
    print("важность групп, с:", res["importance_group_mean_abs_s"])


if __name__ == "__main__":
    main()
