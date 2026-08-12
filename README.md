# FARR-EVA Research Artifact

FARR-EVA is a research prototype for **post-execution arbitration** among
three completed multi-hop QA trajectories: embedded FLARE, IRCoT, and FARR.
It decomposes candidate answers and traces into claims, measures their support
against retrieved evidence with a frozen reranker and NLI model, and applies a
shared pairwise linear utility to select an answer.

This repository is a compact, auditable release extracted from the research
workspace. It intentionally excludes raw datasets, generated candidate pools,
large model checkpoints, experiment logs, and manuscript files.

## Research question

Can a question-level arbitration layer improve a fixed retrieval-reasoning
expert without using dataset identity, expert identity, gold answers, gold
supporting facts, or runtime counters as inference features?

## Method

```mermaid
flowchart LR
    Q[Question and retrieved context] --> F[Embedded FLARE trajectory]
    Q --> I[IRCoT trajectory]
    Q --> R[FARR trajectory]
    F --> V[Claim decomposition and evidence measurement]
    I --> V
    R --> V
    V --> U[Shared pairwise utility]
    U --> A[Anchored arbitration]
    A --> O[Final answer]
```

The selector receives a 28-dimensional evidence vector for every candidate.
The same feature function and utility are shared across candidates. FARR is the
anchor; the selector switches only when the best alternative clears the frozen
utility threshold.

## Verified Test-C result

The frozen FARR-EVA artifact was evaluated once on a question-disjoint Test-C
containing 6,000 questions (2,000 each from HotpotQA, 2WikiMultiHopQA, and
MuSiQue).

| System | Macro F1 | Macro EM |
|---|---:|---:|
| Fixed FARR anchor | 0.5140 | 0.4125 |
| FARR-EVA | 0.5754 | 0.4608 |

The paired macro-F1 difference was **+0.0614**, with a 95% dataset-stratified
bootstrap interval of **[0.0529, 0.0700]**.

These numbers establish improvement over the fixed FARR anchor on Test-C; they
do **not** establish that FARR-EVA is the best possible selector. A later
post-hoc diagnostic applied a corrected earlier portable selector to the same
Test-C candidates and obtained 0.5844 macro F1, 0.0090 above FARR-EVA. Because
Test-C had already been inspected, that comparison is diagnostic rather than
confirmatory. A new balanced fresh Test-D (3,000 questions per dataset) was
locked to resolve the selector comparison and is not reported here before it
is completed.

Machine-readable values and interpretation boundaries are in
[`results/test_c_summary.json`](results/test_c_summary.json) and
[`docs/research_status.md`](docs/research_status.md).

## Install

```bash
cd farr-eva-release
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Candidate generation uses local Hugging Face causal and sequence-classification
models and therefore requires sufficient model storage and, for the reported
configuration, CUDA hardware. Unit tests and the selector demo run without
downloading those models.

## Quick checks

```bash
python -m unittest discover -s tests -v
python examples/selector_demo.py
python scripts/preflight_public_release.py
```

The demo loads the small validation-locked selector artifact and scores a
synthetic three-candidate feature group. It is an API smoke test, not a QA
quality benchmark.

## Reproduction map

| Stage | Entry point |
|---|---|
| Benchmark candidate generation | `run_benchmark.py` |
| Test-C candidate generation | `run_farr_eva_test_c_candidates.py` |
| Evidence-vector extraction | `extract_candidate_evidence_features.py` |
| Pairwise selector training | `train_farr_eva_selector.py` |
| Locked Test-C analysis | `analyze_farr_eva_test_c.py` |

The original datasets are not redistributed. Obtain HotpotQA,
2WikiMultiHopQA, and MuSiQue from their official sources and comply with their
respective licenses. The full candidate pools are also omitted because they
contain generated text and are hundreds of megabytes in size.

## Repository scope

Included:

- retrieval/reasoning expert implementations;
- evidence-vector extraction and arbitration code;
- the small validation-locked linear selector;
- feature and source manifests;
- unit tests and a synthetic demo;
- a compact, provenance-aware result summary.

Excluded:

- raw or prepared datasets;
- generated candidate/evidence rows;
- multi-gigabyte Hugging Face model weights;
- logs, exploratory outputs, manuscripts, and author metadata;
- the unpublished Test-D question manifest and results.

## Limitations

- All reported main benchmarks are English Wikipedia-style multi-hop QA.
- Arbitration pays the cost of producing all required candidate trajectories.
- The evidence measurements depend on frozen reranker and NLI models.
- Candidate-set failures cannot be fixed by selection alone.
- Test-C does not settle the comparison between FARR-EVA and the corrected
  portable selector; the fresh Test-D was created for that purpose.

## License

No open-source license has been selected yet. Public visibility does not grant
permission to copy, modify, or redistribute the code. The author should choose
a license after verifying compatibility with all included code and model
artifacts.
