from __future__ import annotations

import json
from pathlib import Path

from .encoder import build_encoder
from .knowledge import DefenseKnowledge
from .router import ConsensusRouter
from .schemas import LabeledSample


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_samples(path: str | Path) -> list[LabeledSample]:
    samples: list[LabeledSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            samples.append(LabeledSample(text=value["text"], label=value["label"]))
    return samples


def load_knowledge(path: str | Path) -> dict[str, DefenseKnowledge]:
    return {label: DefenseKnowledge.from_dict(value) for label, value in read_json(path).items()}


def save_artifacts(
    directory: str | Path,
    router: ConsensusRouter,
    samples: list[LabeledSample],
    knowledge: dict[str, DefenseKnowledge],
    prompts: dict[str, str],
    fallback_prompt: str,
    encoder_config: dict,
) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "router.json", router.to_state())
    write_json(root / "encoder.json", encoder_config)
    write_json(root / "samples.json", {"items": [vars(sample) for sample in samples]})
    write_json(root / "knowledge.json", {key: vars(value) for key, value in knowledge.items()})
    write_json(root / "prompts.json", prompts)
    (root / "generic_fallback.txt").write_text(fallback_prompt, encoding="utf-8")


def load_artifacts(directory: str | Path):
    root = Path(directory)
    encoder_config = read_json(root / "encoder.json")
    encoder = build_encoder(encoder_config)
    router = ConsensusRouter(encoder).load_state(read_json(root / "router.json"))
    samples = [LabeledSample(**item) for item in read_json(root / "samples.json")["items"]]
    knowledge = {key: DefenseKnowledge.from_dict(value) for key, value in read_json(root / "knowledge.json").items()}
    prompts = read_json(root / "prompts.json")
    fallback = (root / "generic_fallback.txt").read_text(encoding="utf-8")
    return encoder, router, samples, knowledge, prompts, fallback
