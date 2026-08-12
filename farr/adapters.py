from __future__ import annotations

import os
from typing import Any, Dict, List


DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LocalHFLLM:
    """Deterministic Transformers adapter for a single visible CUDA device."""

    def __init__(
        self,
        model_name: str | None = None,
        max_input_tokens: int = 8192,
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run with CUDA_VISIBLE_DEVICES=0 and a CUDA-enabled environment."
            )

        self.model_name = model_name or os.getenv("HF_MODEL") or DEFAULT_LOCAL_MODEL
        self.max_input_tokens = max_input_tokens
        print(f"Loading LLM: {self.model_name}")
        print(f"Visible CUDA device: {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _max_new_tokens(prompt: str) -> int:
        if "[FARR:ANSWER]" in prompt:
            return 64
        if "[FARR:QUERY]" in prompt or "[FARR:VERIFY_QUERY]" in prompt:
            return 160
        return 256

    def __call__(self, prompt: str) -> str:
        import torch

        encoded = self._encode(prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens(prompt),
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion = generated[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def _encode(self, prompt: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise retrieval-augmented QA component. "
                    "Follow the requested output schema exactly and ground every fact in evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{messages[0]['content']}\n\n{prompt}"

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def generate_with_confidence(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Greedy generation with probability for every selected token."""
        import torch

        encoded = self._encode(prompt)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = encoded["input_ids"].shape[1]
        generated_ids = output.sequences[0, input_length:]
        probabilities: List[float] = []
        pieces: List[str] = []
        for token_id, logits in zip(generated_ids, output.scores):
            probability = torch.softmax(logits[0].float(), dim=-1)[token_id].item()
            probabilities.append(float(probability))
            pieces.append(self.tokenizer.decode([token_id], skip_special_tokens=True))

        return {
            "text": self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
            "tokens": pieces,
            "probabilities": probabilities,
        }


class OpenAIChatLLM:
    """Optional API adapter with the same callable interface."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL")
        if not self.model:
            raise ValueError("Set OPENAI_MODEL or pass model=...")
        self.client = OpenAI()

    def __call__(self, prompt: str) -> str:
        response: Any = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise retrieval-augmented QA component.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
