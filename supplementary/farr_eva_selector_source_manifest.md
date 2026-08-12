# FARR-EVA selector-comparator source manifest

This manifest freezes the implementation details needed to reproduce the
post-hoc matched LLM judge, same-feature MLP, and evidence-vector construction.
Hashes are SHA-256 over the named source files on 2026-08-06. The canonical
numeric feature/profile mapping is
`supplementary/farr_eva_feature_schema.json`.

## Matched zero-shot LLM judge

- Runner and prompt/serializer/parser:
  `run_farr_eva_llm_judge_shard.py`, SHA-256
  `4150821dfdba528c5876dd23a7f0b400c1b9abe1cbdfb143593207459ed21327`.
- Aggregation and paired statistics:
  `summarize_farr_eva_llm_judge.py`, SHA-256
  `ff9c5173456edc4c1e60f04f7860ee9c9c7317ad6911875633764929de930d21`.
- Shared local generation adapter dependency:
  `farr_project/farr/adapters.py`, SHA-256
  `990725c00ecb9622cdc38dccd68164b5bccb6281d8e1f30c4fe8820b177c4e84`.
- Model: `Qwen/Qwen2.5-7B-Instruct`, FP16, maximum input 8,192 tokens.
  The judge tag does not match the adapter's answer/query special cases, so
  output generation uses the adapter default of 256 new tokens.
- Decoding: greedy (`do_sample=False`), no temperature/top-p/top-k arguments,
  cache enabled, tokenizer EOS used for EOS and (when necessary) padding.
- System message: `You are a precise retrieval-augmented QA component. Follow
  the requested output schema exactly and ground every fact in evidence.`
- Candidate order is not a pseudorandom shuffle. For every method it computes
  SHA-256 of
  `farr-eva-judge-v1:<dataset>:<question_id>:<method>` and sorts the three
  methods by the hexadecimal digest. Labels A/B/C follow that order. Thus no
  random seed is involved and the mapping is reproducible per question.

The exact user-prompt template is:

```text
[FARR_EVA:ZERO_SHOT_MATCHED_JUDGE_V1]
You must select the best answer to one hard multi-hop question from three
completed candidates. Use only the question and the candidate material below.
You do not know the gold answer, dataset name, supporting-fact labels, expert
identity, or downstream evaluation score.

Judge factual support, contradiction, whether the reasoning connects the
necessary hops, and whether the final answer matches the requested answer
type. Do not reward length or polish. If candidates give the same answer,
choose the candidate with the clearest support.

Return JSON only:
{"selected": "A|B|C", "confidence": 0.0, "reason": "brief reason"}

Question:
<question>

<three serialized candidate blocks>
```

Each candidate block serializes its anonymous label; answer (320-character
limit); retrieval queries (900-character limit); reasoning trace
(1,300-character limit); ten named evidence-derived scalar diagnostics; and up
to four claim-audit entries containing claim text (430-character limit), best
evidence title (160-character limit), entailment, and contradiction. Strings
are whitespace-collapsed and suffix-marked when truncated. Full raw passages,
method identity, dataset name, gold answer, target F1, and supporting-fact
labels are absent.

The requested JSON object has fields `selected`, `confidence`, and `reason`.
Parsing proceeds in four fixed tiers: (1) parse the whole output as JSON and
accept A/B/C found in `selected`; (2) if that fails, parse the first
non-greedy `{...}` substring and apply the same label check; (3) recover A/B/C
with case-insensitive regex
`(?:candidate|selected|answer)\s*[:=]?\s*([ABC])\b`; or (4) deterministically
fall back to FARR. Confidence is converted to float and clipped to [0,1]; a
conversion failure gives zero. On Test-C the tiers account for 5,965, 2, 33,
and 0 outputs, respectively.

## Same-feature MLP

- Source: `run_farr_eva_same_feature_mlp.py`, SHA-256
  `13a176aa733f1c36aee2153471255006eb0446f71d399ea867663da1305179ae`.
- Input: exactly the 28 numeric features and candidate ordering stored in the
  frozen linear artifact. Training-feature mean and population standard
  deviation are fitted on training candidates only and reused unchanged for
  validation and Test-C; standard deviations below `1e-8` are replaced by 1.
- Chosen architecture: `Linear(28,64) -> GELU -> Dropout(0.1) -> Linear(64,1)`.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-5`, batch size 256.
- Pairwise loss for each unequal-F1 unordered pair is
  `softplus(-(u_winner-u_loser))`. Its raw weight is absolute F1 difference
  divided by the number of informative pairs in that dataset, followed by a
  global rescaling to mean one. The batch loss is the weight-normalized mean.
- Gradient norm is clipped to 5.0 before each optimizer step.
- Validation grid: widths {16,32,64}, dropout {0,0.1}, weight decay
  {1e-5,1e-4}; at most 40 epochs with patience 6. Grid training uses seed 42.
  At each epoch the twelve switch thresholds in the manuscript are evaluated.
  The ordering is: satisfy minimum per-dataset delta >= -0.002, maximize macro
  delta, maximize minimum dataset delta, then minimize switch rate. The best
  epoch/state within each grid run defines that configuration; the same rule
  chooses the global configuration. The selected epoch count and threshold
  are then fixed, and the model is retrained from initialization for seeds
  13, 29, 42, 73, and 101. Each saved checkpoint contains the final state,
  feature order, training mean/std, selected configuration, and seed.
- Python, NumPy, and PyTorch seeds are all set; CUDA seeds are set when
  available; deterministic PyTorch algorithms are requested with warnings
  rather than hard failure.

## Evidence-vector and profile sources

- Evidence units, claim extraction, lexical prefilter, candidate aggregation,
  28-feature extraction, schema, and vectorization:
  `farr_star/evidence_verifier.py` functions `evidence_units`,
  `candidate_claims`, `lexical_prefilter`, `_aggregate_candidate`,
  `extract_question_features`, `feature_schema`, and `feature_vector`; SHA-256
  `ebb3df6902bfe58549035682a2c2d4821a4207088fd9a47efa3129670825affb`.
- Question contract and compliance:
  `farr_star/contracts.py` functions `heuristic_contract` and
  `answer_contract_compliance`; SHA-256
  `c3f7f155be06c79396ca1d5a19bc31b0b208000ebac98e2a566886d42a8a7dd0`.
- Profile membership:
  `train_farr_eva_selector.py` function `feature_profiles`; SHA-256
  `57985967c5026128e11b118598fc327f795c15e278628191710c30f5f7b864ad`.
- Selector choice logic:
  `farr_star/eva_selector.py`; SHA-256
  `b6d662466528e6b9c6cda23d8b3b16ec4a386c83e71b579ba62170e3a34d7fbf`.
- Development feature report:
  `data/farr_eva_features/development_report.json`; SHA-256
  `0a2eff003f71e09a74a243d5acb461419e33bd67e375ae93a29b0590219b3aa8`.
- Frozen selector lock:
  `artifacts/farr_eva_v1.lock.json`; SHA-256
  `d9d159d73e6fbbf0e8625265c90a6d43aa354e99e509b1e10303301116fcf62b`.
