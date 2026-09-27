"""Кодек протокола NDTP по спецификации эмулятора (docs/Emulator-and-Telematic-Packets-Specification.md).

Кадр: ``[NPL 15 байт][NPH 10 байт][тело]``, всё little-endian, packed.
CRC-16/Modbus считается по NPH + телу и кладётся в NPL со свапнутыми байтами.

Модуль умеет:

* разбирать поток байт из TCP в кадры (:class:`FrameDecoder`), проверяя сигнатуру и CRC;
* разбирать тело realtime-пакета в ячейки и декодировать навигационную ячейку
  ``G6CellNav00`` в :class:`NavRecord` (то же, что строка ``traffic.csv``);
* собирать кадры (handshake и realtime) — для реплеера исторических данных и тестов.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

NPL_SIGNATURE = 0x7E7E
NPL = struct.Struct("<HHHHBIH")  # signature, dataSize, flags, crc, type, peerAddress, requestId
NPH = struct.Struct("<HHHI")  # serviceId, type, flags, requestId
HANDSHAKE = struct.Struct("<HHHIII")  # protoHigh, protoLow, flags, peerAddress, maxPacketSize, reserved
NAV00 = struct.Struct("<IIIBBHHHHHBB")  # 26 байт

NPL_TYPE_NPH = 0x02
SERVICE_GENERIC_CONTROLS = 0
SERVICE_NAVDATA = 1
NPH_SGC_CONN_REQUEST = 100
NPH_SND_REALTIME = 101

#: Размер полезной нагрузки ячеек по типу (для пропуска неинтересных ячеек).
CELL_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}


def crc16_modbus(data: bytes) -> int:
    """CRC-16/Modbus: poly 0xA001 (отражённый 0x8005), init 0xFFFF."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _swap16(x: int) -> int:
    return ((x & 0xFF) << 8) | (x >> 8)


@dataclass
class Frame:
    """Разобранный кадр NDTP."""

    peer_address: int
    service_id: int
    nph_type: int
    request_id: int
    body: bytes


@dataclass
class NavRecord:
    """Навигационная запись (ячейка ``G6CellNav00``) — аналог строки ``traffic.csv``."""

    unit_id: int
    timestamp: int
    lon: float
    lat: float
    valid: bool
    speed: float
    speed_max: float
    course: float
    altitude: float
    nsat: int
    pdop: int
    flags: dict = field(default_factory=dict)


class FrameDecoder:
    """Инкрементальный разборщик TCP-потока в кадры NDTP.

    Устойчив к мусору и разрывам: ищет сигнатуру ``0x7E7E``, при неверном CRC
    сдвигается на байт и продолжает поиск.
    """

    def __init__(self, check_crc: bool = True):
        self.buf = bytearray()
        self.check_crc = check_crc
        self.crc_errors = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buf.extend(data)
        frames: list[Frame] = []
        while True:
            i = self.buf.find(b"\x7e\x7e")
            if i < 0:
                del self.buf[:-1]
                return frames
            if i:
                del self.buf[:i]
            if len(self.buf) < NPL.size:
                return frames
            sig, size, _flags, crc, typ, peer, _req = NPL.unpack_from(self.buf, 0)
            if len(self.buf) < NPL.size + size:
                return frames
            payload = bytes(self.buf[NPL.size : NPL.size + size])
            # по спецификации CRC лежит в NPL со свапнутыми байтами; прямой порядок тоже
            # принимаем (устойчивость к реализациям терминалов), остальное — мусор
            calc = crc16_modbus(payload) if self.check_crc else None
            if self.check_crc and calc not in (_swap16(crc), crc):
                self.crc_errors += 1
                del self.buf[:1]
                continue
            del self.buf[: NPL.size + size]
            if typ != NPL_TYPE_NPH or size < NPH.size:
                continue
            service, nph_type, _nflags, req = NPH.unpack_from(payload, 0)
            frames.append(Frame(peer, service, nph_type, req, payload[NPH.size :]))


def parse_cells(body: bytes) -> list[tuple[int, int, bytes]]:
    """Режет тело realtime-пакета на ячейки ``(type, number, payload)``.

    Ячейки неизвестного размера обрывают разбор (``G6CellNav00`` всегда первая).
    """
    out = []
    i = 0
    while i + 2 <= len(body):
        typ, num = body[i], body[i + 1]
        size = CELL_SIZES.get(typ)
        if size is None or i + 2 + size > len(body):
            break
        out.append((typ, num, body[i + 2 : i + 2 + size]))
        i += 2 + size
    return out


def decode_nav00(unit_id: int, payload: bytes) -> NavRecord:
    """Декодирует ``G6CellNav00`` (знак координат — из битов N/S и E/W)."""
    ts, lon, lat, bits, _bat, spd, spd_max, course, _track, alt, nsat, pdop = NAV00.unpack(payload)
    north = bool(bits >> 5 & 1)
    east = bool(bits >> 6 & 1)
    return NavRecord(
        unit_id=unit_id,
        timestamp=ts,
        lon=(lon / 1e7) * (1 if east else -1),
        lat=(lat / 1e7) * (1 if north else -1),
        valid=bool(bits >> 7 & 1),
        speed=float(spd),
        speed_max=float(spd_max),
        course=float(course),
        altitude=float(alt),
        nsat=nsat,
        pdop=pdop,
        flags={"alarm": bool(bits >> 1 & 1), "sos": bool(bits >> 2 & 1), "battery": bool(bits >> 4 & 1)},
    )


def nav_records(frame: Frame) -> list[NavRecord]:
    """Все навигационные записи из realtime-кадра."""
    if frame.service_id != SERVICE_NAVDATA or frame.nph_type != NPH_SND_REALTIME:
        return []
    return [decode_nav00(frame.peer_address, p) for t, _n, p in parse_cells(frame.body) if t == 0]


# ----------------------------- сборка кадров -----------------------------


def build_frame(peer: int, service: int, nph_type: int, request_id: int, body: bytes) -> bytes:
    payload = NPH.pack(service, nph_type, 1, request_id & 0xFFFFFFFF) + body
    crc = _swap16(crc16_modbus(payload))
    return NPL.pack(NPL_SIGNATURE, len(payload), 0, crc, NPL_TYPE_NPH, peer, 0) + payload


def build_handshake(unit_id: int, request_id: int = 1) -> bytes:
    body = HANDSHAKE.pack(6, 2, 0, unit_id, 65535, 0)
    return build_frame(unit_id, SERVICE_GENERIC_CONTROLS, NPH_SGC_CONN_REQUEST, request_id, body)


def build_nav00(ts: int, lon: float, lat: float, valid: bool, speed: float = 0, course: float = 0, alt: float = 0) -> bytes:
    bits = (1 << 5 if lat >= 0 else 0) | (1 << 6 if lon >= 0 else 0) | (1 << 7 if valid else 0)
    payload = NAV00.pack(
        int(ts),
        int(round(abs(lon) * 1e7)),
        int(round(abs(lat) * 1e7)),
        bits,
        200,
        int(max(0, min(speed, 65535))),
        int(max(0, min(speed, 65535))),
        int(course) % 361,
        0,
        int(max(0, min(alt, 65535))),
        12,
        10,
    )
    return bytes([0, 0]) + payload


def build_realtime(unit_id: int, request_id: int, cells: bytes) -> bytes:
    return build_frame(unit_id, SERVICE_NAVDATA, NPH_SND_REALTIME, request_id, cells)
