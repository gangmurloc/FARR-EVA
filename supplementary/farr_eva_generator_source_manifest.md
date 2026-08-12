# FARR-EVA candidate-generator source manifest

This manifest identifies the executable source used for the locked Test-C
candidate generation.  The corresponding files, with these exact hashes,
should be bundled with the anonymized reproducibility supplement.

| Source | SHA-256 |
|---|---|
| `farr/adapters.py` | `990725c00ecb9622cdc38dccd68164b5bccb6281d8e1f30c4fe8820b177c4e84` |
| `farr/retrievers.py` | `7bca0be0c7d184f7b847e4bb0c3b92c62afe1c42b4176a3fe5ddf518fcf01844` |
| `farr/baselines.py` | `8b8bcfcf20e8dc33172f2b7228ba1b144d4355ef844dbf8e7158130e963d1ea1` |
| `farr/pipeline_v2.py` | `40cb6550127e6dbad7fe7862a216482b891ec5781603268760e053eddf801950` |
| `farr/prompts.py` | `b9b1e44fdcce82431676c6b2d339602c70759f998a529ed111163e510f66eb6e` |
| `farr/prompts_v2.py` | `97623f4047e911dcb2f8ac76d0449d5e943101628696360bc8f75c3ab9951a43` |
| `run_farr_eva_test_c_candidates.py` | `e1942077a0db00853921b43f4adb15c1dbcb0639e368ef9bb07a44f871cbfd98` |

The `farr/` paths above are repository-relative paths in the anonymized source
supplement; no machine-local path is part of the scientific interface.

## Frozen generator configuration

```text
model = Qwen/Qwen2.5-7B-Instruct
max_input_tokens = 8192
dtype = float16
decoding = greedy (do_sample=False)
initial_top_k = 6
max_hops = 4
max_queries_per_hop = 3
per_query_top_k = 4
hop_evidence_top_k = 5
max_evidence_docs = 16
verification_top_k = 5
max_revision_rounds = 2
max_chars_per_doc = 1000
revise_on_labels = (UNSUPPORTED, UNCERTAIN)
flare_confidence_threshold = 0.20
flare_max_steps = 3
fusion_evidence_top_k = 8
candidate_selector_path = None
```

## Prompt inventory

The prompt text is executable source rather than prose reconstructed after the
experiment.

| System / stage | Source symbol or literal tag |
|---|---|
| Shared system instruction and decoding | `farr/adapters.py: LocalHFLLM` |
| Direct RAG | `farr/baselines.py: _direct_answer_prompt`, tag `[RAG:ANSWER]` |
| FLARE prediction and grounding | `farr/baselines.py: flare`, tags `[FLARE:PREDICT]`, `[FLARE:GROUND]` |
| IRCoT step | `farr/baselines.py: ircot`, tag `[IRCOT:STEP]` |
| RARR verification/revision | `farr/prompts.py: verification_query_prompt`, `verify_prompt`, `revise_prompt` |
| Shared final answer | `farr/prompts.py: answer_prompt`, tag `[FARR:ANSWER]` |
| FARR dispute query and audit | `farr/prompts_v2.py: disagreement_query_prompt`, `candidate_audit_prompt` |
| FARR correction verification | `farr/prompts_v2.py: corrected_answer_verify_prompt` |

The manuscript-facing method name FARR is implemented by `FARRV2` in the
locked source.  This class name is a historical software identifier and is not
a separate manuscript method.
