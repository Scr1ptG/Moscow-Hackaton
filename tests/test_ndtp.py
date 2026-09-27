"""Тесты кодека NDTP (без датасета): кадры, CRC-16/Modbus, устойчивость к мусору и разрезанию потока."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ndtp.protocol import (  # noqa: E402
    FrameDecoder,
    build_handshake,
    build_nav00,
    build_realtime,
    crc16_modbus,
    nav_records,
)


def test_crc_reference_vector():
    assert crc16_modbus(b"123456789") == 0x4B37  # эталон CRC-16/MODBUS


def test_roundtrip_with_garbage_and_split_stream():
    hs = build_handshake(1166336)
    rt = build_realtime(1166336, 2, build_nav00(1767670500, 37.617321, 55.7551234, True, speed=33, course=245, alt=160))
    data = b"junk" + hs + b"\x7e\x7e\x00" + rt
    dec = FrameDecoder()
    frames = []
    for i in range(0, len(data), 5):  # поток режется на куски как в TCP
        frames += dec.feed(data[i : i + 5])
    assert [(f.service_id, f.nph_type) for f in frames] == [(0, 100), (1, 101)]
    (r,) = nav_records(frames[1])
    assert r.unit_id == 1166336 and r.valid and abs(r.lon - 37.617321) < 1e-6 and abs(r.lat - 55.7551234) < 1e-6
    assert r.speed == 33 and r.course == 245


def test_crc_byte_order_tolerance_and_rejects_corruption():
    fr = bytearray(build_realtime(7, 3, build_nav00(1767670500, 37.6, 55.7, True)))
    raw = bytearray(fr)
    crc = struct.unpack_from("<H", raw, 6)[0]
    struct.pack_into("<H", raw, 6, ((crc & 0xFF) << 8) | (crc >> 8))  # «несвапнутый» CRC
    bad = bytearray(fr)
    bad[-1] ^= 0xFF  # испорченное тело
    dec = FrameDecoder()
    assert len(dec.feed(bytes(fr))) == 1
    assert len(dec.feed(bytes(raw))) == 1
    assert len(dec.feed(bytes(bad))) == 0 and dec.crc_errors >= 1


def test_negative_hemispheres():
    fr = build_realtime(1, 1, build_nav00(1, -70.5, -33.4, True))
    (r,) = nav_records(FrameDecoder().feed(fr)[0])
    assert r.lon < 0 and r.lat < 0
