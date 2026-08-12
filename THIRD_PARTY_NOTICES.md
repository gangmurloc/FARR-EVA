# Third-party notices and provenance

This repository combines project-local research code with ideas, public
benchmarks, Python packages, and model interfaces from prior work. It does not
redistribute benchmark records or Hugging Face model weights.

## Prior methods

- **FLARE**: Jiang et al., *Active Retrieval Augmented Generation*.
  The code in this repository contains a local FLARE-inspired adaptation for
  short multi-hop QA; it is not represented as the official FLARE codebase.
- **IRCoT**: Trivedi et al., *Interleaving Retrieval with Chain-of-Thought
  Reasoning for Knowledge-Intensive Multi-Step Questions*.
  The code contains a local IRCoT-style implementation; it is not represented
  as the official IRCoT codebase.

## Benchmarks

HotpotQA, 2WikiMultiHopQA, and MuSiQue are loaded from separately obtained
public releases. Their records are excluded from this repository. Users are
responsible for complying with each dataset's license and terms.

## Models and packages

The reported pipeline references Qwen2.5-7B-Instruct, BGE reranker v2 m3, and
DeBERTa-v3-base-mnli-fever-anli through Hugging Face model identifiers. Their
weights are excluded and remain governed by their respective licenses. Python
dependencies remain governed by their own package licenses.

## Repository license status

No open-source license has yet been granted for the project-local code in this
repository. Public visibility alone does not grant permission to copy, modify,
or redistribute it. A future project license must be selected only after the
provenance and compatibility of every included source file have been reviewed.
