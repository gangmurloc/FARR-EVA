# Reproducibility notes

## Environment snapshot

The direct package versions recorded with the reported local environment are
listed in `requirements-snapshot.txt`. In particular, the timed configuration used
PyTorch 2.5.1 with CUDA 12.1, Transformers 4.44.2, scikit-learn 1.7.2,
NumPy 2.2.6, and SciPy 1.15.3. GPU compatibility still depends on the host
driver and the PyTorch wheel source.

## Frozen models used for evidence measurement

- Reranker: `BAAI/bge-reranker-v2-m3`
- NLI: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- Main generator: `Qwen/Qwen2.5-7B-Instruct`

The exact evidence feature schema and source-file manifests are stored under
`supplementary/`. The small selector artifact under `artifacts/` contains the
validation-selected linear model and scaler.

## Data

This repository does not redistribute benchmark records. The preparation
scripts load public benchmark releases and select question IDs deterministically.
Users must obtain and use each dataset under its own terms.

## Reproduction levels

### Level 1: public artifact smoke test

This level is fully runnable from the repository and verifies artifact loading,
feature dimensionality, and candidate arbitration:

```bash
python examples/selector_demo.py
```

### Level 2: pipeline execution on user-prepared data

The CLI entry points expose the candidate generation, evidence extraction,
training, and analysis stages. Run each entry point with `--help` for the
current argument contract. This level requires the public benchmark releases
and locally prepared split manifests.

### Level 3: exact Test-C regeneration

The exact Test-C question manifest and generated candidate/evidence rows are
not distributed in this compact portfolio release. Consequently, exact Test-C
regeneration is not claimed. The frozen artifact, hashes, source manifests,
machine-readable result summary, and analysis code are provided for audit.

## Expected local outputs

The generation and extraction scripts write JSONL files under user-specified
paths. Keep them outside Git by using the ignored `data/`, `outputs/`, and
`analysis_logs/` directories.

## Statistical reporting

The compact public summary reports paired, dataset-stratified percentile
bootstrap intervals. Exploratory diagnostics must remain labelled separately
from locked confirmation results.
