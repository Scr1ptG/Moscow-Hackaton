/* Дашборд диспетчера: карта ТС и маршрутов, риски, инциденты, объяснения, what-if, метрики.
   Работает с backend (/api/v1/*): по умолчанию тот же origin, иначе ?api=http://host:8000 */
"use strict";

const API = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
const RISK = {
  green: { icon: "●", text: "Норма", cls: "green" },
  yellow: { icon: "▲", text: "Внимание", cls: "yellow" },
  red: { icon: "■", text: "Критично", cls: "red" },
  unknown: { icon: "○", text: "Нет прогноза", cls: "unknown" },
};
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const riskColor = (r) => ({ green: css("--good"), yellow: css("--warning"), red: css("--critical") }[r] || css("--muted"));

const state = { vehicles: new Map(), markers: new Map(), alerts: [], groups: {}, selected: null, overlay: null, ws: null };

// ------------------------------------------------------------------ утилиты
async function api(path, opts = {}) {
  const r = await fetch(API + "/api/v1" + path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
function fmtDelay(s) {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  const sign = s >= 0 ? "+" : "−", a = Math.round(Math.abs(s));
  return `${sign}${Math.floor(a / 60)}:${String(a % 60).padStart(2, "0")}`;
}
function fmtTime(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleString("ru-RU", { timeZone: "Europe/Moscow", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}
const pct = (p) => (p === null || p === undefined ? "—" : `${Math.round(p * 100)}%`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function badge(risk) {
  const r = RISK[risk] || RISK.unknown;
  return `<span class="badge ${r.cls}">${r.icon} ${r.text}</span>`;
}

// ------------------------------------------------------------------ карта
const map = L.map("map", { zoomControl: true }).setView([55.75, 37.62], 10);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18, attribution: "&copy; участники OpenStreetMap",
}).addTo(map);

async function loadRoutes() {
  const routes = await api("/routes");
  const layer = L.layerGroup().addTo(map);
  const pts = [];
  for (const r of routes) {
    const latlngs = r.line.map(([lon, lat]) => [lat, lon]);
    pts.push(...latlngs);
    L.polyline(latlngs, { color: css("--muted"), weight: 2, opacity: 0.75 })
      .bindTooltip(`Маршрут ТС ${r.tr_id}`, { sticky: true }).addTo(layer);
  }
  if (pts.length) {
    state.routeBounds = L.latLngBounds(pts).pad(0.05);
    map.invalidateSize();
    map.fitBounds(state.routeBounds);
  }
}
// контейнер карты может получить размер позже (скрытая вкладка, перестроение вёрстки):
// пересчитываем размер и, если карта ещё «на весь мир», подгоняем под маршруты
new ResizeObserver(() => {
  map.invalidateSize();
  if (state.routeBounds && map.getZoom() < 5) map.fitBounds(state.routeBounds);
}).observe(document.getElementById("map"));

async function ensureGroups() {
  if (Object.keys(state.groups).length) return;
  try {
    const card = await api("/model");
    state.modelCard = card;
    for (const [k, v] of Object.entries(card.groups || {})) state.groups[k] = v;
  } catch (e) { /* ML ещё поднимается */ }
}

function vehicleTooltip(v) {
  const r = RISK[v.risk] || RISK.unknown;
  const conn = v.connection === "lost" ? "<br>✕ нет связи — последнее известное состояние" : "";
  const pred = v.delay_pred !== null && v.delay_pred !== undefined
    ? `<br>прогноз ${fmtDelay(v.delay_pred)} · P(опоздание&gt;2 мин) ${pct(v.p_late)}` : "";
  return `<b>ТС ${v.tr_id}</b> · ${r.icon} ${r.text}${pred}${v.cause ? `<br>${esc(v.cause)}` : ""}${conn}`;
}

function drawVehicles(list) {
  for (const v of list) {
    state.vehicles.set(v.tr_id, v);
    if (v.lat === null || v.lon === null || v.lat === undefined) continue;
    const lost = v.connection === "lost";
    const style = {
      radius: v.risk === "red" ? 9 : 7,
      fillColor: lost ? css("--muted") : riskColor(v.risk),
      fillOpacity: lost ? 0.25 : 0.95,
      color: lost ? css("--muted") : css("--surface"), // кольцо цвета поверхности
      weight: 2, dashArray: lost ? "3,3" : null,
    };
    let m = state.markers.get(v.tr_id);
    if (!m) {
      m = L.circleMarker([v.lat, v.lon], style).addTo(map);
      m.on("click", () => selectVehicle(v.tr_id));
      state.markers.set(v.tr_id, m);
    } else {
      m.setLatLng([v.lat, v.lon]).setStyle(style);
    }
    m.bindTooltip(vehicleTooltip(v));
    if (v.risk === "red") m.bringToFront();
  }
  renderVehiclesTable();
}

// ------------------------------------------------------------------ инциденты
function renderAlerts() {
  const ul = document.getElementById("alerts");
  document.getElementById("alerts-count").textContent = state.alerts.length;
  document.getElementById("alerts-empty").style.display = state.alerts.length ? "none" : "block";
  ul.innerHTML = state.alerts.map((a) => `
    <li class="alert ${a.severity}" data-id="${a.id}">
      <div class="row"><span class="veh">ТС ${a.tr_id}</span>${badge(a.severity)}</div>
      <div class="row"><span>Прогноз: <span class="delay">${fmtDelay(a.delay_pred)}</span> · P ${pct(a.p_late)}</span>
        <span class="meta">${a.status === "acknowledged" ? "✓ принят" : "новый"}</span></div>
      <div class="meta">${esc(a.cause || "")}</div>
      <div class="meta">Цель: ${esc(a.target_address || "—")} · обновлён ${fmtTime(a.updated_at)}</div>
    </li>`).join("");
  ul.querySelectorAll(".alert").forEach((el) => el.addEventListener("click", () => selectAlert(el.dataset.id)));
}

async function refreshAlerts() {
  state.alerts = await api("/alerts");
  renderAlerts();
}

// ------------------------------------------------------------------ карточка
function contribChart(contribs, base, rule) {
  const rows = Object.entries(contribs).map(([k, v]) => [state.groups[k] || k, v]);
  if (rule && Math.abs(rule) >= 0.5) rows.push(["Правило: старт рейса по расписанию", rule]);
  rows.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(1, ...rows.map((r) => Math.abs(r[1])));
  const bar = ([label, v]) => {
    const w = (Math.abs(v) / max) * 36; // запас справа/слева под подпись значения
    const pos = v >= 0;
    const barCss = pos ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
    const valCss = pos ? `left:calc(50% + ${w}% + 4px)` : `right:calc(50% + ${w}% + 4px)`;
    return `<div class="lbl">${esc(label)}</div>
      <div class="track" title="${esc(label)}: ${fmtDelay(v)} (${pos ? "к опозданию" : "к опережению"})">
        <div class="axis"></div><div class="bar ${pos ? "pos" : "neg"}" style="${barCss}"></div>
        <span class="val" style="${valCss}">${fmtDelay(v)}</span></div>`;
  };
  return `<div class="muted" style="font-size:12px;margin-bottom:4px">Обычно ${fmtDelay(base)}; красное — к опозданию, синее — к опережению</div>
    <div class="contrib">${rows.filter((r) => Math.abs(r[1]) >= 1).map(bar).join("")}</div>`;
}

function predictionBlock(p, v) {
  if (!p) return `<p class="hint">По ТС пока нет прогноза (нет плановой остановки в окне 10–15 мин или нет данных).</p>`;
  const top = (p.top_features || []).map((f) =>
    `<tr><td>${esc(f.title)}</td><td>${esc(f.value_text ?? f.value ?? "—")}</td><td>${fmtDelay(f.contrib_s)}</td></tr>`).join("");
  const patterns = (p.patterns || []).map((x) => `<li>${esc(x)}</li>`).join("");
  return `
    <div class="hero"><span class="num">${fmtDelay(p.delay_pred)}</span><span class="unit">мин:с на целевой остановке</span></div>
    <dl class="kv">
      <dt>80%-интервал</dt><dd>${fmtDelay(p.q10)} … ${fmtDelay(p.q90)}</dd>
      <dt>P(опоздание &gt; 2 мин)</dt><dd>${pct(p.p_late)}</dd>
      <dt>Ожидаемая ошибка</dt><dd>±${Math.round(p.expected_abs_error ?? 0)} с</dd>
      <dt>Целевая остановка</dt><dd>${p.target_stop_id ?? "—"} · план ${fmtTime(p.target_plan)}</dd>
      <dt>Текущее отклонение</dt><dd>${fmtDelay(p.cur_dev_s)}</dd>
      <dt>Прогноз построен</dt><dd>${fmtTime(p.T)}${p.stale ? " · <b>устарел (ML недоступен)</b>" : ""}</dd>
    </dl>
    <div class="explain">${esc(p.explanation || "")}</div>
    <h3>Почему такой прогноз (точное разложение)</h3>
    ${contribChart(p.contributions || Object.fromEntries(Object.entries(p).filter(([k]) => k.startsWith("contrib_")).map(([k, x]) => [k.slice(8), x])), p.base_value, p.rule_contrib)}
    ${top ? `<h3>Главные признаки</h3><table class="table"><thead><tr><th>Признак</th><th>Значение</th><th>Вклад</th></tr></thead><tbody>${top}</tbody></table>` : ""}
    ${patterns ? `<h3>Паттерны перед сбоем</h3><ul class="patterns">${patterns}</ul>` : ""}
    <h3>What-if: что если…</h3>
    <div class="whatif">
      <button class="btn ghost" data-w='{"delay_shift_s":120}'>+2 мин опоздания</button>
      <button class="btn ghost" data-w='{"delay_shift_s":-120}'>нагонит 2 мин</button>
      <button class="btn ghost" data-w='{"extra_layover_min":3}'>+3 мин отстоя</button>
      <button class="btn ghost" data-w='{"reserve_vehicle":true}'>резервное ТС по графику</button>
    </div>
    <div class="whatif-out" id="whatif-out"></div>`;
}

function bindWhatIf(trId) {
  document.querySelectorAll("[data-w]").forEach((b) => b.addEventListener("click", async () => {
    const out = document.getElementById("whatif-out");
    out.textContent = "Считаю…";
    try {
      const r = await api("/whatif", { method: "POST", body: JSON.stringify({ tr_id: trId, ...JSON.parse(b.dataset.w) }) });
      out.innerHTML = `Было <b>${fmtDelay(r.before.delay_pred)}</b> (${pct(r.before.p_late)}) → стало <b>${fmtDelay(r.after.delay_pred)}</b>
        (${pct(r.after.p_late)}), изменение ${fmtDelay(r.delta_s)}.<div class="muted" style="margin-top:4px">${esc(r.after.explanation || "")}</div>`;
    } catch (e) { out.textContent = "Не удалось: " + e.message; }
  }));
}

function showOverlay(segment, target, risk) {
  if (state.overlay) state.overlay.remove();
  state.overlay = L.layerGroup().addTo(map);
  const latlngs = (segment || []).map(([lon, lat]) => [lat, lon]);
  if (latlngs.length > 1) L.polyline(latlngs, { color: riskColor(risk), weight: 6, opacity: 0.85 }).addTo(state.overlay);
  if (target) {
    L.circleMarker([target.lat, target.lon], { radius: 8, color: css("--ink"), weight: 2, fillColor: css("--surface"), fillOpacity: 1 })
      .bindTooltip(`Целевая остановка: ${esc(target.address)}<br>план ${fmtTime(target.plan)}`, { permanent: false }).addTo(state.overlay);
    latlngs.push([target.lat, target.lon]);
  }
  if (latlngs.length) map.fitBounds(L.latLngBounds(latlngs).pad(0.3), { maxZoom: 15 });
}

async function selectAlert(id) {
  await ensureGroups();
  const a = await api(`/alerts/${id}`);
  state.selected = { kind: "alert", id };
  const p = { ...a.prediction, contributions: a.contributions };
  document.getElementById("card").innerHTML = `
    <div class="row" style="display:flex;justify-content:space-between;align-items:center">
      <h2 style="margin:0;font-size:16px">Инцидент · ТС ${a.tr_id}</h2>${badge(a.severity)}</div>
    <div class="meta muted">Открыт ${fmtTime(a.opened_at)} · пик прогноза ${fmtDelay(a.peak_delay_s)} · статус: ${a.status === "acknowledged" ? "принят" : a.status === "resolved" ? "закрыт" : "новый"}</div>
    <p><b>Участок:</b> до остановки «${esc(a.target?.address || "—")}»</p>
    ${a.status === "open" ? `<button class="btn" id="btn-ack">Принять в работу</button>` : ""}
    ${predictionBlock(p)}`;
  document.getElementById("card-empty").style.display = "none";
  const ack = document.getElementById("btn-ack");
  if (ack) ack.addEventListener("click", async () => { await api(`/alerts/${id}/ack`, { method: "POST" }); await refreshAlerts(); selectAlert(id); });
  bindWhatIf(a.tr_id);
  showOverlay(a.segment, a.target, a.severity);
  switchTab("card");
}

async function selectVehicle(trId) {
  await ensureGroups();
  const v = await api(`/vehicles/${trId}`);
  if (v.alert_id) return selectAlert(v.alert_id);
  state.selected = { kind: "vehicle", id: trId };
  document.getElementById("card").innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h2 style="margin:0;font-size:16px">ТС ${v.tr_id}</h2>${badge(v.risk)}</div>
    <div class="meta muted">Связь: ${v.connection === "lost" ? "✕ потеряна" : "есть"} · последний пакет ${fmtTime(v.last_t)} · скорость ${v.speed ?? "—"} км/ч</div>
    ${predictionBlock(v.prediction, v)}`;
  document.getElementById("card-empty").style.display = "none";
  if (v.prediction) bindWhatIf(v.tr_id);
  if (state.overlay) state.overlay.remove();
  if (v.lat) map.setView([v.lat, v.lon], 14);
  switchTab("card");
}

// ------------------------------------------------------------------ таблица ТС и метрики
function renderVehiclesTable() {
  const order = { red: 0, yellow: 1, green: 2, unknown: 3 };
  const rows = [...state.vehicles.values()].filter((v) => v.scheduled)
    .sort((a, b) => (order[a.risk] ?? 4) - (order[b.risk] ?? 4) || (b.delay_pred ?? -1e9) - (a.delay_pred ?? -1e9));
  document.querySelector("#vehicles-table tbody").innerHTML = rows.map((v) => `
    <tr class="click" data-tr="${v.tr_id}"><td>${v.tr_id}</td><td>${badge(v.risk)}</td><td>${fmtDelay(v.delay_pred)}</td>
    <td>${pct(v.p_late)}</td><td>${v.connection === "lost" ? "✕ нет" : "✓"}</td></tr>`).join("");
  document.querySelectorAll("#vehicles-table tr.click").forEach((tr) => tr.addEventListener("click", () => selectVehicle(+tr.dataset.tr)));
}

async function refreshMetrics() {
  const [m, h] = await Promise.all([api("/metrics"), api("/health")]);
  const q = m.quality_live || {};
  const gain = q.n ? Math.round((1 - q.mae_model_s / q.mae_cur_dev_s) * 100) : null;
  let model = "";
  try {
    const card = state.modelCard || (state.modelCard = await api("/model"));
    const fi = card.feature_info || {};
    const feats = (card.features || []).map((f) => `<li>${esc(fi[f]?.title || f)}${fi[f]?.monotone ? (fi[f].monotone > 0 ? " ↑" : " ↓") : ""}</li>`).join("");
    const cv = card.cv_report_compact || {}, ex = card.explainability || {};
    model = `<h3>Модель</h3>
      <p class="muted">1 LightGBM, ${card.features?.length ?? "—"} признаков, монотонность (↑/↓), точное SHAP-объяснение.
      Честная оценка (новый маршрут): MAE ${cv.mae_real?.toFixed?.(1) ?? "—"} с. Точность разложения: ${ex.additivity_max_abs_error_s ? ex.additivity_max_abs_error_s.toExponential(0) : "—"} с.</p>
      <ul class="patterns">${feats}</ul>`;
  } catch (e) { model = ""; }
  document.getElementById("metrics").innerHTML = `
    <h3 style="margin-top:0">Качество онлайн (реплей дня с известным фактом)</h3>
    <div class="tiles">
      <div class="tile"><div class="label">MAE модели</div><div class="value">${q.n ? q.mae_model_s + " с" : "—"}</div>
        <div class="delta">${q.n ? `на ${q.n} прогнозах` : "появится, когда цели прогнозов наступят"}</div></div>
      <div class="tile"><div class="label">MAE бейзлайна cur_dev_s</div><div class="value">${q.n ? q.mae_cur_dev_s + " с" : "—"}</div>
        <div class="delta">${gain !== null ? `модель лучше на ${gain}%` : ""}</div></div>
      <div class="tile"><div class="label">Задержка ML (p50 / p95)</div><div class="value">${m.ml_latency?.p50_ms ?? "—"} мс</div>
        <div class="delta">p95 ${m.ml_latency?.p95_ms ?? "—"} мс на весь парк</div></div>
      <div class="tile"><div class="label">Поток</div><div class="value">${h.received.toLocaleString("ru-RU")}</div>
        <div class="delta">пакетов · циклов ${h.cycles} · в буфере ${h.buffered}</div></div>
    </div>
    <p class="muted" style="font-size:12px">${esc(q.note || "")}</p>
    ${model}`;
}

function renderHealth(h) {
  document.getElementById("chip-time").textContent = "⏱ " + fmtTime(h.stream_now) + " МСК";
  const ml = document.getElementById("chip-ml");
  ml.textContent = h.status === "ok" ? "ML: ✓ работает" : `ML: ✕ ${h.ml.breaker} — последнее известное состояние`;
  ml.classList.toggle("bad", h.status !== "ok");
  document.getElementById("chip-stream").textContent = `📡 ${h.received.toLocaleString("ru-RU")} пакетов · ${h.cycles} циклов · ${h.open_alerts} инцид.`;
}

// ------------------------------------------------------------------ живые обновления
function connectWS() {
  const base = API || location.origin;
  const url = base.replace(/^http/, "ws") + "/api/v1/ws";
  try { state.ws = new WebSocket(url); } catch (e) { return; }
  state.ws.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    if (e.type === "cycle" || e.type === "hello") drawVehicles(e.vehicles || []);
    if (e.type.startsWith("alert_")) refreshAlerts();
  };
  state.ws.onclose = () => setTimeout(connectWS, 3000);
}

async function poll() {
  try {
    const h = await api("/health");
    renderHealth(h);
    if (!state.ws || state.ws.readyState !== 1) drawVehicles(await api("/vehicles"));
    await refreshAlerts();
  } catch (e) {
    document.getElementById("chip-ml").textContent = "backend недоступен";
    document.getElementById("chip-ml").classList.add("bad");
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "metrics") refreshMetrics().catch(() => {});
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

document.getElementById("btn-replay").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  btn.disabled = true;
  try {
    await api("/admin/replay", { method: "POST", body: JSON.stringify({ split: "test", speed: +document.getElementById("replay-speed").value }) });
    btn.textContent = "▶ Реплей идёт";
  } catch (e) { alert("Не удалось запустить реплей: " + e.message); btn.disabled = false; }
});

(async function init() {
  await ensureGroups();
  await loadRoutes().catch(() => {});
  await poll();
  connectWS();
  setInterval(poll, 3000);
  setInterval(() => { if (document.getElementById("tab-metrics").classList.contains("active")) refreshMetrics().catch(() => {}); }, 5000);
})();
