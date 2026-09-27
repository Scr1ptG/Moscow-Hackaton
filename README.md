# Предиктор изменений в графике транспорта

ИИ-система раннего предупреждения диспетчера: по потоковой телеметрии NDTP прогнозирует
отклонение от графика (сек) на первой остановке с плановым прибытием через **10–15 мин**,
оценивает вероятность опоздания, объясняет причину и показывает инциденты на карте.
Хакатон Московского транспорта, 2026.

```
 NDTP (эмулятор / реплей датасета) ──TCP:9201──► ingest ──► backend :8000 ◄──► ml-service :8001
                                                              │  REST + WebSocket
                                                              ▼
                                                     dashboard :8080 (карта, инциденты, what-if, метрики)
```

| Модуль | Что делает | Где |
|---|---|---|
| **ML-ядро** | признаки из потока, модель (LightGBM, 9 признаков, монотонность), точные объяснения, калибровка, дообучение | `ml/`, `services/ml_service/` |
| **Backend** | приём телеметрии, цикл прогнозов, инциденты, what-if, живые метрики, деградация без падения | `backend/` |
| **BI-дашборд** | карта маршрутов и ТС с цветом риска, карточки инцидентов с объяснением и рекомендациями, проблемные участки, what-if, метрики | `dashboard/` |
| Приём NDTP | кодек NPL/NPH/CRC-16, TCP-сервер, реплеер исторических данных | `ndtp/` |

## Запуск

```bash
cp .env.example .env              # указать DATA_DIR — путь к датасету (train/ test/ validate/)
docker compose up --build         # дашборд: http://localhost:8080
```
Без Docker: `pip install -r requirements-ml.txt && python run_local.py --replay 120` → http://localhost:8000.
Подробно, в т.ч. как подать поток с эмулятора NDTP, — **[`JURY.md`](JURY.md)**.

## Результаты

| | Значение |
|---|---|
| Лидерборд Data Science | **1.00000** |
| MAE на невиданных маршрутах / «на будущем» (честная валидация) | **68.1 / 65.7 с** против 88.4 / 83.8 у бейзлайна `cur_dev_s` |
| Онлайн-прогноз точки с объяснением | **1.1 мс** (CPU, 1 поток); 500 ТС — 1.13 с на ядро |
| Приём NDTP | ~41 тыс. записей/с на ядро |
| Обрыв связи на 60 мин | MAE 72.6 с — сервис работает по последнему состоянию |

Документы: [`MODEL_CARD.md`](MODEL_CARD.md) (модель и объяснимость) · [`SCALING.md`](SCALING.md)
(дообучение, масштабирование, надёжность) · [`REPORT.md`](REPORT.md) (ход работы, проблемы данных) ·
[`docs/html/index.html`](docs/html/index.html) (Sphinx) · [`docs/openapi/`](docs/openapi) (OpenAPI).

## Структура

```
ml/                  ML-ядро (библиотека)
  data.py            загрузка CSV; факт расписания изолирован от признаков
  schedule.py        структура плана: порядок остановок, рейсы, отстой
  telemetry.py       треки + детектор прибытий на остановки по GPS (map matching)
  features.py        86 признаков точки (tr_id, T) строго по данным <= T
  models.py          LightGBM/CatBoost, leave-one-vehicle-out CV
  compact.py         ПРОДОВАЯ модель: 1 LightGBM, 9 признаков, монотонность, калибровка
  explain_eval.py    проверки объяснимости + графики (reports/figures)
  labels.py          разметка из сырых данных АСДУ (для дообучения)
  retrain.py         дообучение full/warm, гейт champion/challenger
  registry.py        реестр версий модели, продвижение, откат
  degradation.py     качество при обрыве связи
  nn.py              PyTorch: GRU по окну телеметрии + MLP, экспорт ONNX
  train.py           CV-отчёт, финальный ансамбль, submission
  predictor.py       рантайм-инференс: прогноз, интервал, P(опоздание), риск, причина
  explain.py         TreeSHAP-вклады -> группы причин + правила-паттерны
  online.py          онлайн-состояние для потока (те же признаки, что офлайн)
  bench.py           замеры latency
ndtp/                протокол NDTP: кодек (CRC-16/Modbus), TCP-сервер, реплеер traffic.csv
services/ml_service  FastAPI ML-сервис (Swagger: /docs)
backend/             FastAPI backend-оркестратор (Swagger: /docs)
dashboard/           BI-дашборд (HTML/JS, Leaflet), nginx-контейнер
run_local.py         запуск всей системы без Docker
tests/test_no_leak.py  доказательство отсутствия утечки: офлайн-признаки == онлайн (данные <= T)
reports/             cv_report.json, cv_oof.csv, latency.json, логи обучения
submissions/         готовые файлы для платформы
```

## Быстрый старт

```bash
pip install -r requirements-train.txt
set MT_DATA_DIR=C:\Users\ibrag\Downloads\dataset         # путь к распакованному датасету
python -m ml.compact --cv --fit --submit --tag compact     # продовая модель: CV + обучение + submission
python -m ml.explain_eval                                   # проверки объяснимости и графики
python -m ml.train --cv --fit --submit --nn --onnx --tag v2  # исследовательский ансамбль
python -m pytest tests/test_no_leak.py -q                   # проверка анти-утечки
python -m ml.bench                                          # latency
```

ML-сервис и поток:

```bash
uvicorn services.ml_service.app:app --port 8001             # Swagger: http://localhost:8001/docs
curl -X POST localhost:8001/v1/schedule -H "Content-Type: application/json" -d "{\"split\": \"test\"}"
python -m ndtp.server --port 9201 --ml http://localhost:8001  # приём NDTP
python -m ndtp.replay --split test --port 9201 --speed 60      # реальный день в NDTP, 60x
curl "localhost:8001/v1/predictions"                          # прогнозы по всем ТС
```

## Модель в двух словах

Продовая модель — компактная (см. `MODEL_CARD.md`); ниже — общий пайплайн и исследовательский ансамбль.

1. **Детектор прибытий по GPS**: первый вход трека в круг 15 м (и 30 м) вокруг остановки, с
   интерполяцией между фиксациями. Калибровка по фактам АСДУ: медианная ошибка ~4 с.
   Прибытие считается известным только с момента GPS-точки, которая его подтвердила.
2. **Признаки** (86): подсказка `cur_dev_s`, свежая задержка и её тренд по GPS-прибытиям,
   «физический» ETA по темпу движения, структура плана до цели (число остановок, отстой на
   конечной, запас времени), состояние телеметрии (скорость, простой, потеря GPS), история
   того же участка в прошлых рейсах сегодня, час суток.
3. **Ансамбль**: LightGBM (L1) ×5 сидов + CatBoost (MAE) ×2 + GRU/MLP ×3, веса по OOF
   (0.55 / 0.15 / 0.30). Для целей «старт рейса» прогноз сжимается к 0 (×0.15).
   В рантайме CatBoost и NN работают через ONNX Runtime.
4. **Честная валидация**: leave-one-vehicle-out. Синтетические ТС — копии реальных на весь
   день, поэтому копии всегда в одном фолде с оригиналом.

| MAE, с (реальные ТС) | ноль | `cur_dev_s` | модель |
|---|---|---|---|
| leave-one-vehicle-out, все | 93.0 | 88.4 | **68.1** |
| только test | 103.3 | 93.4 | **75.5** |

Метрики и сегментный разбор: `reports/cv_report.json`, сводка в `REPORT.md`.

## Backend (оркестратор, отдельный сервис)

`backend/` — FastAPI-сервис, не импортирует ML-код и общается с ML-сервисом по HTTP.
Поток: `NDTP (эмулятор/реплей) → ndtp.server → backend → ml-service`, дашборд читает backend.

```bash
uvicorn services.ml_service.app:app --port 8001                   # ML-сервис
set ML_URL=http://localhost:8001& set SCHEDULE_SPLIT=test
uvicorn backend.main:app --port 8000                              # backend, Swagger: /docs
python -m ndtp.server --port 9201 --sink http://localhost:8000/api/v1/telemetry
python -m ndtp.replay --split test --port 9201 --speed 60         # или POST /api/v1/admin/replay
python tests/e2e_backend.py                                       # сквозной прогон всей системы
```

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/telemetry` | приём батчей от NDTP-сервера |
| GET | `/api/v1/vehicles` | ТС на карте: положение, связь, риск, прогноз, причина |
| GET | `/api/v1/vehicles/{tr_id}` | карточка ТС + история прогнозов + линия маршрута |
| GET | `/api/v1/routes` | линии маршрутов |
| GET | `/api/v1/alerts?status=active\|all\|resolved` | инциденты |
| GET | `/api/v1/alerts/{id}` | карточка инцидента: прогноз, интервал, P(опоздание), объяснение, вклады, участок |
| POST | `/api/v1/alerts/{id}/ack` | диспетчер принял инцидент |
| POST | `/api/v1/whatif` | сценарии: `delay_shift_s`, `extra_layover_min`, `reserve_vehicle`, `overrides` |
| GET | `/api/v1/segments` | проблемные участки: средняя потеря времени на перегонах по фактическим прибытиям |
| GET | `/api/v1/metrics` | живое качество (MAE на реплее дня с фактом) и latency ML |
| GET | `/api/v1/health`, `/api/v1/model` | состояние (breaker, буфер), карточка модели |
| POST | `/api/v1/admin/replay`, `/api/v1/admin/emulator`, `/api/v1/admin/reload-model` | демо и администрирование |
| WS | `/api/v1/ws` | события `cycle`, `alert_opened/updated/resolved`, `degraded` |

Надёжность: circuit breaker к ML; при недоступности ML телеметрия копится в буфере и досылается,
на карте — последнее известное состояние с пометкой; ТС без телеметрии дольше `STALE_AFTER_S` —
«нет связи». Проверено тестами `tests/test_backend.py`.

Сквозной прогон (`tests/e2e_backend.py`, реальный день 04:00–07:30 за 44 с): 16 772 пакета,
15 циклов, 6 инцидентов; живая MAE 65.6 с против 91.9 у `cur_dev_s`; ML — 22 мс на весь парк.
