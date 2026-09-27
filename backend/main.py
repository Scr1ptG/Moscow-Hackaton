"""Точка входа backend: ``uvicorn backend.main:app --port 8000`` (Swagger: /docs)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings
from .ml_client import CircuitBreaker, MLClient
from .schedule import Schedule
from .state import Store
from .worker import Worker


class Hub(set):
    """Подписчики WebSocket; рассылка событий всем, отвалившиеся удаляются."""

    async def broadcast(self, event: dict):
        for ws in list(self):
            try:
                await ws.send_json(event)
            except Exception:  # noqa: BLE001
                self.discard(ws)


def create_app(cfg: Settings | None = None, ml: MLClient | None = None, start_loops: bool = True) -> FastAPI:
    """Фабрика приложения (``ml`` можно подменить в тестах)."""
    cfg = cfg or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.hub = Hub()
        schedule = Schedule.load(cfg.data_dir, cfg.schedule_split)
        client = ml or MLClient(cfg.ml_url, cfg.ml_timeout_s, CircuitBreaker(cfg.breaker_failures, cfg.breaker_cooldown_s))
        app.state.worker = Worker(Store(), schedule, client, cfg, app.state.hub.broadcast)
        if start_loops:
            app.state.worker.start()
        logging.getLogger("backend").info("backend: %d ТС в расписании (%s)", len(schedule.vehicle_ids()), cfg.schedule_split)
        yield
        await app.state.worker.stop()
        if ml is None:
            await client.close()

    app = FastAPI(
        title="Transit Delay Predictor — Backend",
        version="1.0.0",
        description="Оркестрация: приём телеметрии NDTP, прогнозы ML-сервиса, инциденты, API и WebSocket для дашборда.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    # BI-дашборд (статический модуль dashboard/): при локальном запуске его раздаёт backend,
    # в Docker — отдельный контейнер nginx (dashboard/Dockerfile)
    dash = Path(__file__).resolve().parent.parent / "dashboard"
    if dash.exists():
        app.mount("/dashboard", StaticFiles(directory=dash, html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/dashboard/")
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # не логировать каждый запрос к ML
app = create_app()
