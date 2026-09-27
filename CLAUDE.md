# CLAUDE.md — карта проекта (читать первым, код без нужды не перечитывать)

Хакатон Московского транспорта, «Предиктор изменений в графике транспорта»: прогноз
отклонения от графика (сек) на первой остановке с планом в (T+10, T+15] мин.
Дедлайн CSV и формы: **27.09.2026 23:59 МСК**. Команда 5 чел., пользователь — ML.
Отвечать по-русски. Датасет: `C:\Users\ibrag\Downloads\dataset` (env `MT_DATA_DIR`).
Окружение: Windows, Python 3.14 (`pip install --user`), RTX 4060; **нет Docker и Node**.

## Где что
- `ml/compact.py` — **продовая модель**: 1 LightGBM (Huber, 7 листьев × 300), 9 признаков,
  монотонность `intermediate`, правило «старт рейса ×0.15», калибровка по OOF-остаткам.
- `ml/features.py` — 86 признаков строго по данным ≤ T; `ml/telemetry.py` — детектор прибытий
  (вход в круг 15/30 м, маскирование по `arr_known`); `ml/schedule.py` — рейсы/отстой.
- `ml/predictor.py` — `CompactPredictor` (прод, точный SHAP, текст объяснения) и
  `DelayPredictor` (ансамбль); `load_predictor()` по `ML_MODEL=compact|ensemble`.
- `ml/online.py` — онлайн-состояние (numpy, кэш прибытий, шардирование `shard_of` = crc32).
- `ml/labels.py` — разметка из сырых данных АСДУ; `ml/retrain.py` + `ml/registry.py` — дообучение
  (full/warm, гейт champion/challenger, версии, откат); `ml/degradation.py` — обрыв связи.
- `ml/explain_eval.py` — проверки объяснимости + графики; `ml/bench.py` — latency и нагрузка.
- `ml/train.py`, `ml/nn.py`, `ml/models.py` — исследовательский ансамбль (LGB×5+Cat×2+GRU×3).
- `services/ml_service/app.py` — ML-сервис FastAPI :8001; `ndtp/` — кодек, TCP-сервер (`--sink`), реплеер.
- `backend/` — отдельный FastAPI :8000 (без импорта ml): приём телеметрии, цикл прогнозов по «времени
  потока» (`ml_synced_t`), инциденты, what-if, живые метрики, WS; тесты `tests/test_backend.py`.
- Документы: `MODEL_CARD.md`, `SCALING.md`, `REPORT.md`, `PLAN.md`, `README.md`.

## Команды
```
python -m ml.compact --cv --fit --submit --tag X   # продовая модель
python -m ml.explain_eval                           # объяснимость -> reports/explainability.json
python -m ml.retrain --simulate | --new-data DIR --mode warm|full | --list | --rollback
python -m ml.bench ; python -m ml.degradation
python -m pytest tests/test_no_leak.py -q           # должен проходить после любых правок признаков
python tests/e2e_stream.py                          # NDTP -> ML-сервис -> прогнозы
python tests/e2e_backend.py                         # NDTP -> backend -> ML: вся система
python -m pytest tests/ -q                          # no-leak + backend
```

## Решения и факты (не переоткрывать)
- Утечки ответа validate: факты в `train|test/schedule.csv`; `manual_fill` в validate plan. **Не используем.**
- Синтетические ТС (9000000+) — копии реальных на весь день со сдвигом и шумом σ≈20 с →
  валидация **только leave-one-vehicle-out**; модели с 86 признаками «узнают» копии.
- `cur_dev_s` = отклонение на последней остановке по ПЛАНУ ≤ T; цель — первая с планом в окне.
- После отстоя на конечной задержка обнуляется; старт рейса — шум около 0.
- LightGBM + ONNX Runtime в одном процессе: только `num_threads=1` и без spin-wait (иначе ×30 latency).
- LightGBM: L1 несовместим с монотонностью; `advanced` даёт нарушения → `intermediate`.

## Честный протокол (ml/protocol.py) — решать только по нему
- P1 leave-one-vehicle-out (новый маршрут) и P3 `forward_in_time` (новое время) — решающие.
- P2 train→test нечестен: разметка того же маршрута после T; копии синтетики — утечка.
- Финальное обучение: `twin_contaminated` исключает копии validate-периодов (±30 мин).
- Компактная: P1 68.2, P3 65.7; LightGBM-86: 69.2 / 68.2. 12 гипотез отклонены (MODEL_CARD).

## Метрики (MAE, с; LOVO, реальные ТС)
ноль 93.0 · `cur_dev_s` 88.4 · компактная **68.1** (test 74.1) · ансамбль 68.05 (test 75.5).
Лидерборд: ансамбль v3 = **1.00000** (лучший, в зачёте); компактная = 0.81415
(разница из-за синтетических копий validate в train — см. `REPORT.md`).
Онлайн: 1.1 мс на точку, 500 ТС — 1.1 с на ядро, 4 шарда — 0.28 с.
