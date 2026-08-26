from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_query: str) -> str: ...


class MockLLMBackend:
    """Safe deterministic backend for smoke tests; not an experimental model."""

    def generate(self, system_prompt: str, user_query: str) -> str:
        return json.dumps(
            {
                "recovered_request": user_query,
                "risk_label": "BENIGN",
                "final_response": f"Processed safely: {user_query}",
            },
            ensure_ascii=False,
        )


@dataclass
class TransformersLLMBackend:
    """Local Hugging Face causal-LM backend using the model chat template."""

    model_name: str
    revision: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install evoshield[models] to use local target models") from exc
        dtype = self.torch_dtype
        if dtype != "auto":
            dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def generate(self, system_prompt: str, user_query: str) -> str:
        import torch

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = f"{system_prompt}\n\n{user_query}".strip()
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_backend(config: dict) -> LLMBackend:
    value = dict(config)
    backend_type = value.pop("type", "mock")
    if backend_type == "mock":
        return MockLLMBackend()
    if backend_type == "transformers":
        return TransformersLLMBackend(**value)
    raise ValueError(f"Unknown backend type: {backend_type}")
