"""Backend-оркестратор «Предиктора изменений в графике транспорта».

Отдельный сервис (FastAPI), не импортирует ML-код: общается с ML-сервисом по HTTP.

* :mod:`backend.config` — настройки из переменных окружения;
* :mod:`backend.schedule` — расписание, остановки и маршруты для карты;
* :mod:`backend.ml_client` — клиент ML-сервиса с circuit breaker;
* :mod:`backend.state` — состояние ТС, прогнозов, алертов, метрик;
* :mod:`backend.alerts` — жизненный цикл инцидентов;
* :mod:`backend.worker` — фоновые циклы: пересылка телеметрии и прогнозы;
* :mod:`backend.api` / :mod:`backend.main` — REST + WebSocket для дашборда.
"""
