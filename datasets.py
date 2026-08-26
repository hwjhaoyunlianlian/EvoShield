from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AttackRecord:
    sample_id: str
    base_request_id: str
    base_request: str
    attack_prompt: str
    attack_type: str
    split: str
    target_model: str = "shared"
    attack_generator: str = "provided"

    @classmethod
    def from_dict(cls, value: dict) -> "AttackRecord":
        prompt = value.get("attack_prompt", value.get("text", ""))
        label = value.get("attack_type", value.get("label", ""))
        return cls(
            sample_id=str(value.get("sample_id", value.get("id", ""))),
            base_request_id=str(value.get("base_request_id", value.get("harmful_id", ""))),
            base_request=str(value.get("base_request", value.get("harmful", prompt))),
            attack_prompt=str(prompt),
            attack_type=str(label),
            split=str(value.get("split", "unspecified")),
            target_model=str(value.get("target_model", "shared")),
            attack_generator=str(value.get("attack_generator", label or "provided")),
        )


def read_records(path: str | Path) -> list[AttackRecord]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
    else:
        values = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [AttackRecord.from_dict(value) for value in values]


def write_records(path: str | Path, records: Iterable[AttackRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(asdict(record), ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def assert_disjoint(records: Iterable[AttackRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        previous = seen.get(record.base_request_id)
        if previous is not None and previous != record.split:
            raise ValueError(
                f"Base request {record.base_request_id!r} occurs in both {previous!r} and {record.split!r}"
            )
        seen[record.base_request_id] = record.split
