from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import Encoder
from .schemas import LabeledSample, RouteDecision


@dataclass
class RouterConfig:
    alpha: float = 0.8
    k: int = 7
    gamma_c: float = 0.3203
    gamma_p: float = 0.8031
    epsilon: float = 1e-8
    epochs: int = 300
    learning_rate: float = 0.15
    l2: float = 1e-4
    seed: int = 42


class ConsensusRouter:
    """Linear global classifier + similarity-weighted local prototype branch."""

    def __init__(self, encoder: Encoder, config: RouterConfig | None = None):
        self.encoder = encoder
        self.config = config or RouterConfig()
        self.classes: list[str] = []
        self.prototypes = np.empty((0, 0))
        self.prototype_labels: list[str] = []
        self.weights = np.empty((0, 0))
        self.bias = np.empty(0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, samples: list[LabeledSample]) -> "ConsensusRouter":
        if not samples:
            raise ValueError("At least one labeled sample is required")
        self.classes = sorted({sample.label for sample in samples})
        class_to_id = {label: i for i, label in enumerate(self.classes)}
        x = self.encoder.encode([sample.text for sample in samples])
        y = np.array([class_to_id[sample.label] for sample in samples])
        targets = np.eye(len(self.classes))[y]
        rng = np.random.default_rng(self.config.seed)
        self.weights = rng.normal(0, 0.01, (x.shape[1], len(self.classes)))
        self.bias = np.zeros(len(self.classes))
        for _ in range(self.config.epochs):
            probabilities = self._softmax(x @ self.weights + self.bias)
            error = (probabilities - targets) / len(samples)
            grad_w = x.T @ error + self.config.l2 * self.weights
            grad_b = error.sum(axis=0)
            self.weights -= self.config.learning_rate * grad_w
            self.bias -= self.config.learning_rate * grad_b
        self.prototypes = x.copy()
        self.prototype_labels = [sample.label for sample in samples]
        return self

    def _require_fit(self) -> None:
        if not self.classes:
            raise RuntimeError("Router has not been fitted")

    def score(self, query: str) -> RouteDecision:
        self._require_fit()
        vector = self.encoder.encode([query])
        global_values = self._softmax(vector @ self.weights + self.bias)[0]
        similarities = (vector @ self.prototypes.T)[0]
        count = min(self.config.k, len(similarities))
        neighbors = np.argsort(similarities)[::-1][:count]
        local_values = np.zeros(len(self.classes), dtype=np.float64)
        clipped = np.maximum(similarities[neighbors], 0.0) + self.config.epsilon
        denominator = clipped.sum()
        for weight, index in zip(clipped, neighbors):
            class_id = self.classes.index(self.prototype_labels[int(index)])
            local_values[class_id] += weight / denominator
        fused = self.config.alpha * global_values + (1.0 - self.config.alpha) * local_values
        best_id = int(np.argmax(fused))
        best_type = self.classes[best_id]
        prototype_similarity = float(similarities.max())
        accepted = bool(
            fused[best_id] >= self.config.gamma_c
            and prototype_similarity >= self.config.gamma_p
        )
        mapping = lambda values: {name: float(values[i]) for i, name in enumerate(self.classes)}
        return RouteDecision(
            route=best_type if accepted else "Fallback",
            predicted_type=best_type,
            accepted=accepted,
            fused_confidence=float(fused[best_id]),
            prototype_similarity=prototype_similarity,
            global_scores=mapping(global_values),
            local_scores=mapping(local_values),
            fused_scores=mapping(fused),
        )

    def to_state(self) -> dict:
        self._require_fit()
        return {
            "classes": self.classes,
            "prototype_labels": self.prototype_labels,
            "prototypes": self.prototypes.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "config": vars(self.config),
        }

    def load_state(self, state: dict) -> "ConsensusRouter":
        self.classes = list(state["classes"])
        self.prototype_labels = list(state["prototype_labels"])
        self.prototypes = np.asarray(state["prototypes"], dtype=np.float64)
        self.weights = np.asarray(state["weights"], dtype=np.float64)
        self.bias = np.asarray(state["bias"], dtype=np.float64)
        self.config = RouterConfig(**state["config"])
        return self
