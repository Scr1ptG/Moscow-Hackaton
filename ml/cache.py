"""Кэш признаков по сплитам (parquet в ``artifacts/``)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .data import load_split
from .features import build_features

ART_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def get_features(split: str, rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Возвращает таблицу признаков сплита, строя её при необходимости."""
    ART_DIR.mkdir(exist_ok=True)
    path = ART_DIR / f"features_{split}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    t0 = time.time()
    df = build_features(load_split(split))
    df.to_parquet(path, index=False)
    if verbose:
        print(f"[features] {split}: {df.shape} за {time.time() - t0:.1f} c -> {path.name}")
    return df


def get_all(rebuild: bool = False) -> pd.DataFrame:
    """Признаки train+test (с разметкой) и validate в одной таблице."""
    return pd.concat([get_features(s, rebuild) for s in ("train", "test", "validate")], ignore_index=True)


def get_sequences(split: str, rebuild: bool = False, verbose: bool = True):
    """Окна телеметрии для NN (порядок строк = порядок :func:`get_features`)."""
    import numpy as np

    from .features import prepare_context
    from .nn import build_sequences

    ART_DIR.mkdir(exist_ok=True)
    path = ART_DIR / f"seq_{split}.npy"
    if path.exists() and not rebuild:
        return np.load(path)
    t0 = time.time()
    sp = load_split(split)
    seq = build_sequences(sp, prepare_context(sp))
    np.save(path, seq)
    if verbose:
        print(f"[seq] {split}: {seq.shape} за {time.time() - t0:.1f} c -> {path.name}")
    return seq


def get_all_sequences(rebuild: bool = False):
    import numpy as np

    return np.concatenate([get_sequences(s, rebuild) for s in ("train", "test", "validate")])
