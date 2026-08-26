from __future__ import annotations

import json

from .llm import LLMBackend
from .router import ConsensusRouter
from .schemas import DefenseResult


class EvoShieldDefense:
    def __init__(
        self,
        router: ConsensusRouter,
        prompts: dict[str, str],
        fallback_prompt: str,
        backend: LLMBackend,
    ):
        self.router = router
        self.prompts = prompts
        self.fallback_prompt = fallback_prompt
        self.backend = backend

    def defend(self, query: str) -> DefenseResult:
        route = self.router.score(query)
        type_specific = route.accepted and route.route in self.prompts
        prompt = self.prompts[route.route] if type_specific else self.fallback_prompt
        raw = self.backend.generate(prompt, query)
        parsed: dict = {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        return DefenseResult(
            route=route,
            prompt_kind="type_specific" if type_specific else "fallback",
            raw_output=raw,
            recovered_request=parsed.get("recovered_request"),
            risk_label=parsed.get("risk_label"),
            final_response=parsed.get("final_response"),
        )
