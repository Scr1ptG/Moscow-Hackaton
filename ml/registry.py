"""Реестр версий продовой модели: неизменяемые версии, продвижение, откат.

Структура::

    artifacts/registry/<version>/   model.txt, meta.json, report.json  (неизменяемо)
    artifacts/registry/CURRENT      id версии в продакшене
    artifacts/compact/              копия текущей версии — её читает ML-сервис

ML-сервис подхватывает новую версию без рестарта: ``POST /v1/admin/reload``.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .cache import ART_DIR

REGISTRY = ART_DIR / "registry"
PROD_DIR = ART_DIR / "compact"


def save_version(src_dir: Path, report: dict, tag: str = "") -> str:
    """Копирует артефакты модели из ``src_dir`` в новую неизменяемую версию."""
    REGISTRY.mkdir(parents=True, exist_ok=True)
    vid = time.strftime("v%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    dst = REGISTRY / vid
    shutil.copytree(src_dir, dst)
    (dst / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return vid


def current() -> str | None:
    f = REGISTRY / "CURRENT"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def promote(vid: str) -> None:
    """Делает версию продовой: атомарная подмена ``artifacts/compact``."""
    src = REGISTRY / vid
    if not (src / "model.txt").exists():
        raise FileNotFoundError(src)
    tmp = PROD_DIR.with_name("compact.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    old = PROD_DIR.with_name("compact.old")
    if old.exists():
        shutil.rmtree(old)
    if PROD_DIR.exists():
        PROD_DIR.rename(old)
    tmp.rename(PROD_DIR)
    (REGISTRY / "CURRENT").write_text(vid, encoding="utf-8")


def rollback() -> str:
    """Откат на предыдущую версию из реестра."""
    versions = list_versions()
    cur = current()
    prev = [v["version"] for v in versions if v["version"] != cur]
    if not prev:
        raise RuntimeError("нет версии для отката")
    promote(prev[-1])
    return prev[-1]


def list_versions() -> list[dict]:
    if not REGISTRY.exists():
        return []
    out = []
    for d in sorted(p for p in REGISTRY.iterdir() if p.is_dir()):
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
        out.append({"version": d.name, "current": d.name == current(), **{k: rep.get(k) for k in ("mode", "holdout_mae", "decision")}})
    return out
