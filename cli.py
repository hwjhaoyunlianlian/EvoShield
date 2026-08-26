from __future__ import annotations

import argparse
import json
from pathlib import Path

from .defense import EvoShieldDefense
from .encoder import build_encoder
from .knowledge import build_type_specific_prompt
from .llm import build_backend
from .registration import register_attack_type, summarize_pattern
from .repository import load_artifacts, load_knowledge, read_json, read_samples, save_artifacts
from .router import ConsensusRouter, RouterConfig


def build(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    encoder = build_encoder(config["encoder"])
    samples = read_samples(args.registered_data)
    router = ConsensusRouter(encoder, RouterConfig(**config["router"])).fit(samples)
    knowledge = load_knowledge(args.knowledge)
    prompts = {}
    for label in router.classes:
        typed = [sample for sample in samples if sample.label == label]
        prompts[label] = build_type_specific_prompt(label, summarize_pattern(typed), knowledge[label])
    fallback = Path(args.fallback_prompt).read_text(encoding="utf-8")
    save_artifacts(args.output, router, samples, knowledge, prompts, fallback, encoder.config())
    print(json.dumps({"classes": router.classes, "output": args.output}, indent=2))


def route(args: argparse.Namespace) -> None:
    _, router, *_ = load_artifacts(args.artifacts)
    print(json.dumps(router.score(args.query).to_dict(), indent=2, ensure_ascii=False))


def defend(args: argparse.Namespace) -> None:
    _, router, _, _, prompts, fallback = load_artifacts(args.artifacts)
    backend_config = read_json(args.backend_config) if args.backend_config else {"type": "mock"}
    engine = EvoShieldDefense(router, prompts, fallback, build_backend(backend_config))
    print(json.dumps(engine.defend(args.query).to_dict(), indent=2, ensure_ascii=False))


def register(args: argparse.Namespace) -> None:
    encoder, router, samples, knowledge, prompts, fallback = load_artifacts(args.artifacts)
    support = read_samples(args.support_data)
    new_knowledge = load_knowledge(args.knowledge_entry)
    knowledge.update(new_knowledge)
    samples, knowledge, prompts = register_attack_type(router, samples, support, knowledge, prompts)
    encoder_config = encoder.config()
    save_artifacts(args.artifacts, router, samples, knowledge, prompts, fallback, encoder_config)
    print(json.dumps({"registered_type": support[0].label, "support_size": len(support)}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EvoShield reference implementation")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("--config", required=True)
    p.add_argument("--registered-data", required=True)
    p.add_argument("--knowledge", required=True)
    p.add_argument("--fallback-prompt", default="prompts/generic_fallback.txt")
    p.add_argument("--output", default="artifacts")
    p.set_defaults(func=build)
    p = commands.add_parser("route")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--query", required=True)
    p.set_defaults(func=route)
    p = commands.add_parser("defend")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--backend-config")
    p.set_defaults(func=defend)
    p = commands.add_parser("register")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--support-data", required=True)
    p.add_argument("--knowledge-entry", required=True)
    p.set_defaults(func=register)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
