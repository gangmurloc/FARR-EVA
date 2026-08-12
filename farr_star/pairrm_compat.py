from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)


class DebertaV2PairRM(DebertaV2PreTrainedModel):
    """Compatibility copy of the official LLM-Blender PairRM head."""

    def __init__(self, config):
        super().__init__(config)
        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size
        self.sep_token_id = config.sep_token_id
        self.source_prefix_id = config.source_prefix_id
        self.cand_prefix_id = config.cand_prefix_id
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id
        self.head_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.drop_out),
            torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(self.drop_out),
            torch.nn.Linear(self.hidden_size, self.n_tasks),
        )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )
        keep = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep]
        attention_mask = attention_mask[:, keep]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        encodings = outputs.hidden_states[-1]
        source_indexes = torch.where(
            input_ids == self.source_prefix_id
        )
        candidate1_indexes = torch.where(
            input_ids == self.cand1_prefix_id
        )
        candidate2_indexes = torch.where(
            input_ids == self.cand2_prefix_id
        )
        source = encodings[
            source_indexes[0], source_indexes[1], :
        ]
        candidate1 = encodings[
            candidate1_indexes[0], candidate1_indexes[1], :
        ]
        candidate2 = encodings[
            candidate2_indexes[0], candidate2_indexes[1], :
        ]
        left = self.head_layer(torch.cat([source, candidate1], dim=-1))
        right = self.head_layer(
            torch.cat([source, candidate2], dim=-1)
        )
        logits = (left - right).mean(dim=-1)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=(
                outputs.hidden_states if output_hidden_states else None
            ),
            attentions=outputs.attentions,
        )


class PairRMRanker:
    def __init__(
        self,
        model_name: str = "llm-blender/PairRM-hf",
        *,
        local_files_only: bool = True,
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = (
            torch.float16 if self.device.startswith("cuda")
            else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = DebertaV2PairRM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _encode(
        self,
        sources: Sequence[str],
        candidate1: Sequence[str],
        candidate2: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        rows = []
        for source, left, right in zip(
            sources, candidate1, candidate2
        ):
            source_ids = self.tokenizer.encode(
                f"<|source|>{source}",
                max_length=1224,
                truncation=True,
            )
            remaining = 2048 - len(source_ids)
            candidate_length = max(32, remaining // 2)
            left_ids = self.tokenizer.encode(
                f"<|candidate1|>{left}",
                max_length=candidate_length,
                truncation=True,
            )
            right_ids = self.tokenizer.encode(
                f"<|candidate2|>{right}",
                max_length=candidate_length,
                truncation=True,
            )
            rows.append(source_ids + left_ids + right_ids)
        encoded = self.tokenizer.pad(
            {"input_ids": rows},
            return_tensors="pt",
            padding=True,
        )
        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    def compare(
        self,
        sources: Sequence[str],
        candidate1: Sequence[str],
        candidate2: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        logits = []
        with torch.inference_mode():
            for start in range(0, len(sources), batch_size):
                end = start + batch_size
                encoded = self._encode(
                    sources[start:end],
                    candidate1[start:end],
                    candidate2[start:end],
                )
                output = self.model(**encoded)
                logits.extend(
                    output.logits.detach().float().cpu().tolist()
                )
        return np.asarray(logits, dtype=float)

    def rank(
        self,
        sources: Sequence[str],
        candidates: Sequence[Sequence[str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        if not sources:
            return np.empty((0, 0), dtype=int)
        candidate_count = len(candidates[0])
        pairs = list(combinations(range(candidate_count), 2))
        flat_sources = []
        left_values = []
        right_values = []
        owners = []
        for owner, (source, values) in enumerate(
            zip(sources, candidates)
        ):
            if len(values) != candidate_count:
                raise ValueError("Candidate counts must be equal.")
            for left, right in pairs:
                flat_sources.append(source)
                left_values.append(str(values[left]))
                right_values.append(str(values[right]))
                owners.append((owner, left, right))
        logits = self.compare(
            flat_sources,
            left_values,
            right_values,
            batch_size=batch_size,
        )
        scores = np.zeros((len(sources), candidate_count), dtype=float)
        for logit, (owner, left, right) in zip(logits, owners):
            probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            scores[owner, left] += probability
            scores[owner, right] += 1.0 - probability
        ranks = np.empty_like(scores, dtype=int)
        order = np.argsort(-scores, axis=1)
        for index, values in enumerate(order):
            ranks[index, values] = np.arange(1, candidate_count + 1)
        return ranks
