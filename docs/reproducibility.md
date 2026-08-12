# Reproducibility notes

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

## Expected local outputs

The generation and extraction scripts write JSONL files under user-specified
paths. Keep them outside Git by using the ignored `data/`, `outputs/`, and
`analysis_logs/` directories.

## Statistical reporting

The compact public summary reports paired, dataset-stratified percentile
bootstrap intervals. Exploratory diagnostics must remain labelled separately
from locked confirmation results.

