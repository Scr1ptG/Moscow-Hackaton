"""Геометрия на малых расстояниях (локальная равнопромежуточная проекция)."""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_000.0


def haversine(lon1, lat1, lon2, lat2):
    """Расстояние по дуге большого круга, метры (векторизовано)."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_xy(lon, lat, lat0: float = 55.75):
    """Переводит lon/lat в локальные метры (x — восток, y — север)."""
    k = np.pi / 180 * EARTH_R
    return np.asarray(lon) * k * np.cos(np.radians(lat0)), np.asarray(lat) * k


def point_segment(px, py, ax, ay, bx, by):
    """Расстояние от точек P до отрезков AB и параметр проекции ``u`` в [0, 1]."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    u = np.clip(u, 0.0, 1.0)
    qx, qy = ax + u * dx, ay + u * dy
    return np.hypot(px - qx, py - qy), u


def bearing(lon1, lat1, lon2, lat2):
    """Азимут от точки 1 к точке 2, градусы [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_diff(a, b):
    """Модуль разницы углов, градусы [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(d, 360 - d)
