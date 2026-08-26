from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .datasets import read_records
from .defense import EvoShieldDefense
from .evaluators import HarmBenchEvaluator, LLMJudgeEvaluator, RefusalHeuristicEvaluator
from .llm import build_backend
from .metrics import routing_metrics
from .repository import load_artifacts, read_json
from .encoder import build_encoder
from .router import ConsensusRouter, RouterConfig
from .schemas import LabeledSample


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No result rows were produced")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def routing(args: argparse.Namespace) -> None:
    _, router, *_ = load_artifacts(args.artifacts)
    records = read_records(args.data)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for record in records:
        grouped[(record.target_model, record.attack_type)].append((router.score(record.attack_prompt), record.attack_type))
    rows: list[dict] = []
    for (target_model, attack_type), values in sorted(grouped.items()):
        metrics = routing_metrics([item[0] for item in values], [item[1] for item in values])
        rows.append({"target_model": target_model, "attack_type": attack_type, "n": len(values), **asdict(metrics)})
    _write_csv(args.output, rows)


def defense(args: argparse.Namespace) -> None:
    _, router, _, _, prompts, fallback = load_artifacts(args.artifacts)
    backend = build_backend(read_json(args.backend_config))
    if args.judge_type == "heuristic":
        judge = RefusalHeuristicEvaluator()
    elif args.judge_type == "harmbench":
        judge = HarmBenchEvaluator(build_backend(read_json(args.judge_config)))
    else:
        judge = LLMJudgeEvaluator(build_backend(read_json(args.judge_config)))
    engine = EvoShieldDefense(router, prompts, fallback, backend)
    rows = []
    for record in read_records(args.data):
        started = time.perf_counter()
        result = engine.defend(record.attack_prompt)
        elapsed = time.perf_counter() - started
        response = result.final_response or result.raw_output
        rows.append(
            {
                "sample_id": record.sample_id,
                "target_model": record.target_model,
                "attack_type": record.attack_type,
                "route": result.route.route,
                "accepted": result.route.accepted,
                "prompt_kind": result.prompt_kind,
                "unsafe": judge.unsafe(record.base_request, response),
                "latency_seconds": elapsed,
                "recovered_request": result.recovered_request or "",
                "risk_label": result.risk_label or "",
                "response": response,
            }
        )
    _write_csv(args.output, rows)


def summarize(args: argparse.Namespace) -> None:
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for value in values:
        grouped[(value.get("target_model", "shared"), value.get("attack_type", "all"))].append(value)
    rows = []
    for (model, attack), items in sorted(grouped.items()):
        unsafe = [item["unsafe"].lower() == "true" for item in items]
        rows.append({"target_model": model, "attack_type": attack, "n": len(items), "asr": sum(unsafe) / len(unsafe)})
    _write_csv(args.output, rows)


def leave_one_out(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    records = read_records(args.data)
    labels = sorted({record.attack_type for record in records})
    rows = []
    for held_out in labels:
        train = [record for record in records if record.attack_type != held_out]
        unknown = [record for record in records if record.attack_type == held_out]
        router = ConsensusRouter(build_encoder(config["encoder"]), RouterConfig(**config["router"])).fit(
            [LabeledSample(record.attack_prompt, record.attack_type) for record in train]
        )
        unknown_decisions = [router.score(record.attack_prompt) for record in unknown]
        known_decisions = [router.score(record.attack_prompt) for record in train]
        known_metrics = routing_metrics(known_decisions, [record.attack_type for record in train])
        fallback_count = sum(not decision.accepted for decision in unknown_decisions)
        rows.append(
            {
                "held_out_attack": held_out,
                "unknown_n": len(unknown),
                "unregistered_fallback_rate": fallback_count / len(unknown),
                "unregistered_misrouting_rate": 1.0 - fallback_count / len(unknown),
                "known_n": len(train),
                "known_route_coverage": known_metrics.route_coverage,
                "known_selective_accuracy": known_metrics.selective_accuracy,
            }
        )
    _write_csv(args.output, rows)


def registration(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    records = read_records(args.data)
    labels = sorted({record.attack_type for record in records})
    support_sizes = [int(value) for value in args.support_sizes.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for held_out in labels:
        held = [record for record in records if record.attack_type == held_out]
        base = [record for record in records if record.attack_type != held_out]
        for seed in seeds:
            import numpy as np

            rng = np.random.default_rng(seed)
            order = rng.permutation(len(held)).tolist()
            for support_size in support_sizes:
                if support_size >= len(held):
                    continue
                support = [held[index] for index in order[:support_size]]
                test = [held[index] for index in order[support_size:]]
                router_config = dict(config["router"])
                router_config["seed"] = seed
                router = ConsensusRouter(build_encoder(config["encoder"]), RouterConfig(**router_config)).fit(
                    [LabeledSample(record.attack_prompt, record.attack_type) for record in [*base, *support]]
                )
                decisions = [router.score(record.attack_prompt) for record in test]
                registered_accuracy = sum(d.predicted_type == held_out for d in decisions) / len(decisions)
                retention = [router.score(record.attack_prompt) for record in base]
                retention_accuracy = sum(
                    decision.predicted_type == record.attack_type for decision, record in zip(retention, base)
                ) / len(base)
                rows.append(
                    {
                        "held_out_attack": held_out,
                        "support_size": support_size,
                        "seed": seed,
                        "test_n": len(test),
                        "registration_accuracy": registered_accuracy,
                        "retention_accuracy": retention_accuracy,
                    }
                )
    _write_csv(args.output, rows)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EvoShield paper experiment runner")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("routing", help="Evaluate pre-gating and selective routing")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=routing)
    p = commands.add_parser("defense", help="Run end-to-end defense")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--backend-config", required=True)
    p.add_argument("--judge-config")
    p.add_argument("--judge-type", choices=["heuristic", "llm-json", "harmbench"], default="heuristic")
    p.add_argument("--output", required=True)
    p.set_defaults(func=defense)
    p = commands.add_parser("summarize", help="Summarize raw defense outputs into ASR")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=summarize)
    p = commands.add_parser("leave-one-out", help="Reproduce the zero-shot held-out routing protocol")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=leave_one_out)
    p = commands.add_parser("registration", help="Run few-shot held-out attack registration")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--support-sizes", default="1,5,10,20")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--output", required=True)
    p.set_defaults(func=registration)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
