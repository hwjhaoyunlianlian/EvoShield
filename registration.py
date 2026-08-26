from __future__ import annotations

from .knowledge import DefenseKnowledge, build_type_specific_prompt
from .router import ConsensusRouter
from .schemas import LabeledSample


def summarize_pattern(samples: list[LabeledSample]) -> str:
    """Deterministic placeholder analyzer; replace with the paper's analyzer agent."""
    lengths = [len(sample.text.split()) for sample in samples]
    mean_length = sum(lengths) / max(len(lengths), 1)
    return f"Externally identified type represented by {len(samples)} labeled samples; mean length {mean_length:.1f} tokens."


def register_attack_type(
    router: ConsensusRouter,
    registered_samples: list[LabeledSample],
    support_samples: list[LabeledSample],
    knowledge: dict[str, DefenseKnowledge],
    prompts: dict[str, str],
) -> tuple[list[LabeledSample], dict[str, DefenseKnowledge], dict[str, str]]:
    labels = {sample.label for sample in support_samples}
    if len(labels) != 1:
        raise ValueError("A registration support set must contain exactly one labeled attack type")
    label = next(iter(labels))
    if label not in knowledge:
        raise ValueError(f"No six-component knowledge entry supplied for {label!r}")
    merged = [*registered_samples, *support_samples]
    router.fit(merged)
    pattern = summarize_pattern(support_samples)
    prompts[label] = build_type_specific_prompt(label, pattern, knowledge[label])
    return merged, knowledge, prompts
