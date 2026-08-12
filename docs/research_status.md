# Research status and interpretation boundaries

## What is established

On the locked Test-C, FARR-EVA improved over its validation-fixed FARR anchor
by 0.0614 macro F1. The interval excluded zero, and all three dataset-level
differences were positive. The selector artifact and feature profile had been
locked before Test-C inference.

FARR was the validation-fixed anchor and primary comparator, not a reference
chosen after inspecting Test-C. The full fixed-system context is RARR 0.3971,
RAG 0.4227, FLARE 0.5106, embedded FLARE 0.5109, IRCoT 0.5121, and FARR
0.5140 macro F1, compared with FARR-EVA at 0.5754. Thus, FARR was also the
strongest globally fixed single expert by Test-C macro F1. A separate
test-selected per-dataset-best diagnostic uses FARR for HotpotQA, FLARE for
2WikiMultiHopQA, and IRCoT for MuSiQue; FARR-EVA exceeds that composite
reference by 0.0468 [0.0391, 0.0547]. That diagnostic is not the primary
confirmatory comparison because its reference identities were determined from
Test-C aggregates.

## What is not established

The Test-C result does not prove that evidence-vector arbitration is superior
to every alternative representation or selector. After Test-C had been
inspected, a cache bug was found in an earlier portable selector evaluation.
Replaying that corrected selector on Test-C produced 0.5844 macro F1, exceeding
FARR-EVA's 0.5754. This is an important diagnostic, but not an untouched
comparison because the split was no longer unseen.

## Why Test-D exists

A fresh, balanced Test-D was locked before inference with 3,000 questions from
each of HotpotQA, 2WikiMultiHopQA, and MuSiQue. It excludes every question used
in prior training, validation, Test-B, and Test-C manifests. Its primary
comparison is the corrected frozen portable selector against fixed FARR; the
portable-selector versus FARR-EVA comparison is secondary and uses a
predefined outcome rule.

The Test-D manifest is deliberately absent from this release until the run is
complete. Publishing the held-out IDs in advance would weaken the clean
confirmation protocol.

## Public-release policy

This repository reports negative and conflicting evidence rather than
discarding it. Future result updates should preserve the same rule: update the
status regardless of direction or statistical significance, and never retune
either frozen selector on Test-D.
