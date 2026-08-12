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

### Source-provenance audit status

On 2026-08-12, the public project files were compared against the then-current
heads of the [official FLARE repository](https://github.com/jzbjyb/FLARE/tree/ec4b06b502b5ab54f3f9236b0112a5d28482e7bb)
(`ec4b06b5`) and [official IRCoT repository](https://github.com/stonybrooknlp/ircot/tree/3c1820f698eea5eeddb4fba3c56b64c961e063e4)
(`3c1820f6`). No identical project source files or project-specific prompt
markers were found. The local package also uses a different compact module and
runtime structure from both upstream repositories. This is evidence against
wholesale vendoring, but it is **not** a legal certification that every line
was independently implemented. The author must still confirm whether any
individual code fragments were consulted or adapted before selecting a
project-wide open-source license.

Upstream licenses observed during that audit:

- FLARE official repository: MIT License.
- IRCoT official repository: Apache License 2.0.

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
