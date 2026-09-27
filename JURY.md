# Инструкция для жюри

Система из трёх модулей: **ML-ядро** (`ml-service`), **backend** (оркестратор) и **BI-дашборд**
(`dashboard`), плюс приёмник NDTP (`ingest`). Нужен датасет хакатона (папка с `train/ test/ validate/`).

## 1. Запуск

### Вариант А — Docker (основной)
```bash
cp .env.example .env            # в .env указать DATA_DIR=<путь к датасету>
docker compose up --build       # ml-service :8001, backend :8000, dashboard :8080, ingest :9201
```
Открыть **дашборд: http://localhost:8080**.

### Вариант Б — без Docker (одна команда)
```bash
pip install -r requirements-ml.txt
set MT_DATA_DIR=<путь к датасету>          # Linux/macOS: export MT_DATA_DIR=...
python run_local.py --replay 120          # поднимет всё и сразу подаст поток
```
Открыть **дашборд: http://localhost:8000**.

## 2. Как подать поток телеметрии

**Исторический датасет (реальные треки дня, с прогнозами):**
- кнопка **«▶ Запустить»** в шапке дашборда (ускорение ×60/×120/×300), или
- `docker compose --profile replay up replay`, или
- `POST http://localhost:8000/api/v1/admin/replay` с телом `{"split": "test", "speed": 120}`.

Реплеер шлёт строки `traffic.csv` настоящими пакетами NDTP (handshake + `NPH_SND_REALTIME` с
`G6CellNav00`, CRC-16/Modbus) на TCP :9201 — ровно как бортовые терминалы.

**Эмулятор NDTP из раздачи:**
```bash
docker load -i ndtp-telemetry-emulator.tar
docker compose --profile emulator up -d emulator
curl -X POST http://localhost:8000/api/v1/admin/emulator -H "Content-Type: application/json" -d "{\"units\": [1166336, 1166337], \"interval_ms\": 5000}"
```
Эмулятор подключается к `ingest:9201`, проходит handshake и шлёт realtime-пакеты. Его терминалы
генерируют случайную навигацию и не привязаны к расписанию, поэтому на карте они видны как ТС без
прогноза («○»); прогнозы строятся для ТС из расписания (поток из датасета).

## 3. Где смотреть

| Что | Где |
|---|---|
| Карта: маршруты и ТС; уровень риска цветом **на ТС и на всём маршруте** (● норма / ▲ внимание / ■ критично / ✕ нет связи); сводка по парку | дашборд, основной экран |
| Алерты (инциденты) | вкладка «Инциденты»; `GET /api/v1/alerts` |
| Карточка инцидента: прогноз опоздания, 80%-интервал, P(опоздание > 2 мин), причина, текстовое объяснение, **рекомендации диспетчеру**, вклады факторов, паттерны, участок маршрута на карте | клик по инциденту / ТС; `GET /api/v1/alerts/{id}` |
| What-if (доп. опоздание, доп. отстой, резервное ТС) | кнопки в карточке; `POST /api/v1/whatif` |
| **Проблемные участки** — где ТС сегодня систематически теряют время (по фактическим прибытиям из GPS) | переключатель на легенде карты, вкладка «Участки»; `GET /api/v1/segments` |
| Метрики онлайн (MAE модели против бейзлайна на реплее дня с известным фактом), задержка ML, поток | вкладка «Метрики»; `GET /api/v1/metrics` |
| Карточка модели (признаки, монотонность, метрики объяснимости) | вкладка «Метрики»; `GET /api/v1/model` |
| Swagger | backend http://localhost:8000/docs · ML http://localhost:8001/docs |
| Поток событий | WebSocket `ws://localhost:8000/api/v1/ws` |

Факт прибытия из файла используется **только** для отображения метрик качества на реплее прошедшего
дня и никогда не передаётся в модель.

## 4. Документация и отчёты

- Код (Sphinx, HTML): `docs/html/index.html`; OpenAPI: `docs/openapi/backend.json`, `docs/openapi/ml-service.json`.
- Модель и объяснимость: `MODEL_CARD.md` (+ графики `reports/figures/`).
- Дообучение, масштабирование, надёжность: `SCALING.md`.
- Что и как делали, найденные проблемы данных: `REPORT.md`.

## 5. Проверки

```bash
pip install -r requirements-train.txt
python -m pytest tests/ -q          # анти-утечка (офлайн = онлайн) + backend
python tests/e2e_backend.py         # сквозной прогон: NDTP -> backend -> ML
python -m ml.bench                  # latency
```
