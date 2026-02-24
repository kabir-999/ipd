from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_KEYS = ["ndvi", "ndwi", "savi", "evi", "ndmi", "nbr", "vegetationPercent"]


def _find_model_path() -> Path:
    candidates = sorted(MODEL_DIR.glob("*.keras"))
    if not candidates:
        raise HTTPException(status_code=500, detail="No .keras LSTM model found in /models")
    return candidates[0]


@lru_cache(maxsize=1)
def _get_model():
    if load_model is None:
        raise HTTPException(status_code=500, detail="TensorFlow is not installed. Add tensorflow to backend requirements.")

    model_path = _find_model_path()
    try:
        return load_model(model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load LSTM model: {exc}") from exc


def _to_feature_array(features: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for row in features:
        rows.append([float(row.get(key, 0.0)) for key in FEATURE_KEYS])
    return np.array(rows, dtype=np.float32)


def _to_sequence(base: np.ndarray, sequence_length: int, feature_dim: int) -> np.ndarray:
    if feature_dim > base.shape[1]:
        padding = np.zeros((base.shape[0], feature_dim - base.shape[1]), dtype=np.float32)
        base = np.concatenate([base, padding], axis=1)
    elif feature_dim < base.shape[1]:
        base = base[:, :feature_dim]

    if base.shape[0] >= sequence_length:
        seq = base[-sequence_length:]
    else:
        pad = np.zeros((sequence_length - base.shape[0], base.shape[1]), dtype=np.float32)
        seq = np.concatenate([pad, base], axis=0)

    return np.expand_dims(seq, axis=0)


def predict_risk_with_debug(features: list[dict[str, Any]]) -> tuple[float, list[str]]:
    if not features:
        raise HTTPException(status_code=400, detail="No yearly features available for model prediction")

    debug: list[str] = []
    model = _get_model()

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    sequence_length = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else len(features)
    feature_dim = input_shape[2] if len(input_shape) > 2 and input_shape[2] is not None else len(FEATURE_KEYS)

    base = _to_feature_array(features)
    debug.append(f"LSTM input shape (raw yearly array): {base.shape}")
    debug.append(f"LSTM full array: {base.tolist()}")

    if base.shape[0] < int(sequence_length):
        debug.append(
            f"LSTM warning: only {base.shape[0]} timesteps available, model expects {int(sequence_length)}. Padding applied."
        )

    x = _to_sequence(base, int(sequence_length), int(feature_dim))
    debug.append(f"LSTM model tensor shape: {x.shape}")

    try:
        pred = model.predict(x, verbose=0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    risk = float(np.ravel(pred)[-1])

    if risk < 0.0 or risk > 1.0:
        risk = float(1.0 / (1.0 + np.exp(-risk)))

    risk = round(risk, 4)
    debug.append(f"LSTM predicted risk: {risk}")

    return risk, debug


def predict_risk(features: list[dict[str, Any]]) -> float:
    risk, _ = predict_risk_with_debug(features)
    return risk
