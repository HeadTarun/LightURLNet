from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "lighturlnet.onnx"
VOCAB_PATH = ROOT / "vocab.json"
CONFIG_PATH = ROOT / "config.json"


def normalize_url(url: str) -> str:
    value = url.strip().lower()
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
    return value


@lru_cache(maxsize=1)
def load_assets() -> tuple[Any, dict[str, int], int]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocab = json.load(file)["char_to_idx"]

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session, vocab, int(config.get("max_len", 256))


def encode_url(url: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    normalized = normalize_url(url)
    pad_idx = vocab.get("<PAD>", 0)
    unk_idx = vocab.get("<UNK>", 1)
    ids = [vocab.get(char, unk_idx) for char in normalized[:max_len]]
    ids.extend([pad_idx] * (max_len - len(ids)))
    return np.asarray([ids], dtype=np.int64)


def _danger_probability(raw_output: np.ndarray) -> float:
    output = np.asarray(raw_output, dtype=np.float32).squeeze()
    danger_class_index = int(os.getenv("LIGHTURLNET_DANGER_CLASS_INDEX", "1"))

    if output.ndim == 0:
        return float(1.0 / (1.0 + np.exp(-output)))

    if output.size == 1:
        return float(1.0 / (1.0 + np.exp(-output.reshape(-1)[0])))

    logits = output.reshape(-1)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    danger_class_index = max(0, min(danger_class_index, probabilities.size - 1))
    return float(probabilities[danger_class_index])


def predict_url(url: str) -> dict[str, Any]:
    if not url.strip():
        raise ValueError("Enter a URL to scan.")

    session, vocab, max_len = load_assets()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_ids = encode_url(url, vocab, max_len)
    raw = session.run([output_name], {input_name: input_ids})[0]

    danger_probability = _danger_probability(raw)
    label = "Danger" if danger_probability >= 0.5 else "Safe"
    confidence = danger_probability if label == "Danger" else 1.0 - danger_probability

    return {
        "url": normalize_url(url),
        "label": label,
        "danger_probability": danger_probability,
        "safe_probability": 1.0 - danger_probability,
        "confidence": confidence,
    }
