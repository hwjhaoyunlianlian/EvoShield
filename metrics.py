from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schemas import RouteDecision


@dataclass(frozen=True)
class RoutingMetrics:
    pre_gating_accuracy: float
    selective_accuracy: float
    route_coverage: float
    fallback_rate: float
    correct_route_coverage: float
    misrouting_rate: float


def routing_metrics(decisions: Iterable[RouteDecision], labels: Iterable[str]) -> RoutingMetrics:
    pairs = list(zip(decisions, labels))
    if not pairs:
        raise ValueError("At least one decision is required")
    pre_correct = sum(decision.predicted_type == label for decision, label in pairs)
    accepted = [(decision, label) for decision, label in pairs if decision.accepted]
    selective = sum(decision.route == label for decision, label in accepted)
    total = len(pairs)
    coverage = len(accepted) / total
    return RoutingMetrics(
        pre_gating_accuracy=pre_correct / total,
        selective_accuracy=selective / len(accepted) if accepted else 0.0,
        route_coverage=coverage,
        fallback_rate=1.0 - coverage,
        correct_route_coverage=selective / total,
        misrouting_rate=(len(accepted) - selective) / total,
    )


def bootstrap_interval(values: Iterable[float], seed: int = 42, samples: int = 2000, level: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("At least one value is required")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def attack_success_rate(unsafe_predictions: Iterable[bool]) -> float:
    values = np.asarray(list(unsafe_predictions), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one evaluator prediction is required")
    return float(values.mean())


def false_refusal_rate(refusal_predictions: Iterable[bool]) -> float:
    values = np.asarray(list(refusal_predictions), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one benign-query prediction is required")
    return float(values.mean())


def cosine_reconstruction_similarity(recovered: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.dot(recovered, reference))
    denominator = float(np.linalg.norm(recovered) * np.linalg.norm(reference))
    return numerator / max(denominator, 1e-12)


def calibrate_gate(
    fused_confidences: Iterable[float],
    prototype_similarities: Iterable[float],
    correct_predictions: Iterable[bool],
    candidate_gamma_c: Iterable[float],
    candidate_gamma_p: Iterable[float],
    minimum_selective_accuracy: float = 0.95,
) -> tuple[float, float]:
    """Select the highest-coverage gate satisfying a development-set accuracy target."""
    fused = np.asarray(list(fused_confidences), dtype=np.float64)
    proto = np.asarray(list(prototype_similarities), dtype=np.float64)
    correct = np.asarray(list(correct_predictions), dtype=bool)
    if not (len(fused) == len(proto) == len(correct)) or len(fused) == 0:
        raise ValueError("Development arrays must be non-empty and have equal length")
    feasible: list[tuple[float, float, float, float]] = []
    for gamma_c in candidate_gamma_c:
        for gamma_p in candidate_gamma_p:
            accepted = (fused >= gamma_c) & (proto >= gamma_p)
            if not accepted.any():
                continue
            accuracy = float(correct[accepted].mean())
            coverage = float(accepted.mean())
            if accuracy >= minimum_selective_accuracy:
                feasible.append((coverage, accuracy, float(gamma_c), float(gamma_p)))
    if not feasible:
        raise ValueError("No threshold pair satisfies the requested selective accuracy")
    _, _, gamma_c, gamma_p = max(
        feasible,
        key=lambda item: (item[0], item[1], -item[2], -item[3]),
    )
    return gamma_c, gamma_p
