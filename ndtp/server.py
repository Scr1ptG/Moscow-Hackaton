"""NDTP TCP-сервер: принимает поток от эмулятора/реплеера и отдаёт записи в sink.

По умолчанию sink — батчевая отправка в backend (``POST /api/v1/telemetry``)
раз в ``flush_s`` секунд. При недоступности ML-сервиса записи остаются в
буфере (ограниченном), сервер не падает; при обрыве TCP-соединения эмулятор
сам переподключается и повторяет handshake — сервер к этому готов.

Запуск::

    python -m ndtp.server --port 9201 --sink http://localhost:8000/api/v1/telemetry
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import urllib.request
from collections import deque

from .protocol import (
    NPH_SGC_CONN_REQUEST,
    SERVICE_GENERIC_CONTROLS,
    FrameDecoder,
    nav_records,
)

log = logging.getLogger("ndtp.server")


class HttpBatchSink:
    """Копит записи и отправляет батчами в ML-сервис; переживает его недоступность."""

    def __init__(self, url: str, flush_s: float = 1.0, max_buffer: int = 200_000):
        """:param url: полный адрес приёма батчей (backend ``/api/v1/telemetry`` или ML ``/v1/telemetry``)."""
        self.url = url
        self.flush_s = flush_s
        self.buf: deque = deque(maxlen=max_buffer)
        self.sent = 0
        self.errors = 0

    def put(self, rec: dict):
        self.buf.append(rec)

    async def run(self):
        while True:
            await asyncio.sleep(self.flush_s)
            if not self.buf:
                continue
            batch = [self.buf.popleft() for _ in range(min(len(self.buf), 5000))]
            try:
                await asyncio.to_thread(self._post, batch)
                self.sent += len(batch)
            except Exception as e:  # noqa: BLE001 — деградация: вернуть в буфер и жить дальше
                self.errors += 1
                self.buf.extendleft(reversed(batch))
                log.warning("ML-сервис недоступен (%s), в буфере %d записей", e, len(self.buf))

    def _post(self, batch: list[dict]):
        req = urllib.request.Request(self.url, data=json.dumps(batch).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()


class NdtpServer:
    """Asyncio-сервер NDTP: одна корутина на TCP-соединение терминала."""

    def __init__(self, sink, host: str = "0.0.0.0", port: int = 9201):
        self.sink, self.host, self.port = sink, host, port
        self.stats = {"connections": 0, "frames": 0, "nav": 0, "crc_errors": 0}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        self.stats["connections"] += 1
        dec = FrameDecoder()
        try:
            while data := await reader.read(65536):
                for fr in dec.feed(data):
                    self.stats["frames"] += 1
                    if fr.service_id == SERVICE_GENERIC_CONTROLS and fr.nph_type == NPH_SGC_CONN_REQUEST:
                        log.info("handshake unit=%s from %s", fr.peer_address, peer)
                        continue
                    for r in nav_records(fr):
                        self.stats["nav"] += 1
                        self.sink.put(
                            {"unit_id": r.unit_id, "t": r.timestamp, "valid": r.valid, "lon": r.lon, "lat": r.lat,
                             "speed": r.speed, "heading": r.course, "recv": time.time()}
                        )
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.stats["crc_errors"] += dec.crc_errors
            writer.close()

    async def serve(self):
        srv = await asyncio.start_server(self.handle, self.host, self.port)
        log.info("NDTP слушает %s:%s", self.host, self.port)
        async with srv:
            await srv.serve_forever()


async def _main(a):
    sink = HttpBatchSink(a.sink)
    server = NdtpServer(sink, port=a.port)
    await asyncio.gather(server.serve(), sink.run())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9201)
    ap.add_argument("--sink", default="http://localhost:8000/api/v1/telemetry",
                    help="куда слать батчи: backend (по умолчанию) или напрямую ML-сервис .../v1/telemetry")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(_main(ap.parse_args()))
