from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class LabeledSample:
    text: str
    label: str


@dataclass(frozen=True)
class RouteDecision:
    route: str
    predicted_type: str
    accepted: bool
    fused_confidence: float
    prototype_similarity: float
    global_scores: dict[str, float]
    local_scores: dict[str, float]
    fused_scores: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DefenseResult:
    route: RouteDecision
    prompt_kind: Literal["type_specific", "fallback"]
    raw_output: str
    recovered_request: str | None = None
    risk_label: str | None = None
    final_response: str | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["route"] = self.route.to_dict()
        return value


@dataclass(frozen=True)
class ValidationScore:
    safety: float
    semantic: float
    policy: float
    risk: float

    def weighted(self, weights: dict[str, float]) -> float:
        values = asdict(self)
        return sum(float(weights[key]) * values[key] for key in values)
