from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
    def config(self) -> dict[str, Any]: ...


@dataclass
class HashingEncoder:
    """Deterministic dependency-light encoder for demos and tests.

    Replace this class with a pretrained semantic encoder for paper experiments.
    """

    dimension: int = 768
    ngram_min: int = 1
    ngram_max: int = 2

    _token_pattern = re.compile(r"[A-Za-z0-9_]+|[^\W\s]", re.UNICODE)

    def _features(self, text: str) -> list[str]:
        tokens = [token.lower() for token in self._token_pattern.findall(text)]
        features: list[str] = []
        for size in range(self.ngram_min, self.ngram_max + 1):
            features.extend(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                index = number % self.dimension
                sign = 1.0 if (number >> 63) == 0 else -1.0
                matrix[row, index] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def config(self) -> dict[str, Any]:
        return {
            "type": "hashing",
            "dimension": self.dimension,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
        }


class SentenceTransformerEncoder:
    """Paper-grade normalized semantic encoder.

    The dependency is imported lazily so the lightweight smoke test remains usable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        revision: str | None = None,
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install evoshield[models] to use the semantic encoder") from exc
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    def config(self) -> dict[str, Any]:
        return {
            "type": "sentence_transformer",
            "model_name": self.model_name,
            "device": self.device,
            "revision": self.revision,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
        }


def build_encoder(config: dict[str, Any]) -> Encoder:
    value = dict(config)
    encoder_type = value.pop("type", "hashing")
    if encoder_type == "hashing":
        return HashingEncoder(**value)
    if encoder_type == "sentence_transformer":
        return SentenceTransformerEncoder(**value)
    raise ValueError(f"Unknown encoder type: {encoder_type}")
