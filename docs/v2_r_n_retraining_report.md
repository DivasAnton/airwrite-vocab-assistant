# V2 R/N Retraining Report

## Purpose

This report records the baseline before rebuilding the dataset manifest and
training model v2 from scratch. The main goal is to measure whether the added
lowercase `r` variants improve recognition without causing `n` to be predicted
as `r`.

The model has 26 identity classes and does not distinguish uppercase from
lowercase. Both `r` writing styles therefore belong to class `R`.

## Baseline Before Retraining

Source artifacts:

- Metrics: `artifacts/metadata/metrics.json`
- Classification report: `artifacts/reports/classification_report.json`
- Confusion matrix: `artifacts/reports/confusion_matrix.json`
- Error analysis: `artifacts/reports/error_analysis.csv`

| Metric | Current model |
| --- | ---: |
| Test samples | 575 |
| Test accuracy | 80.52% |
| Macro F1 | 80.62% |
| Top-3 accuracy | 92.17% |
| `R` precision | 80.95% |
| `R` recall | 77.27% |
| `R` F1 | 79.07% |
| `R` test support | 22 |
| `N` precision | 71.43% |
| `N` recall | 68.18% |
| `N` F1 | 69.77% |
| `N` test support | 22 |

These values are the comparison baseline. Do not overwrite this section after
training; add the new result below it.

## New Data Definition

| Group | Model label | Intended content | Target count |
| --- | --- | --- | ---: |
| `r_1` | `R` | Clear lowercase `r`, writing style 1 |  |
| `r_2` | `R` | Lowercase `r` variant that resembles `n` |  |
| `n` | `N` | `n` samples, including hard negatives close to `r_2` |  |

The two `r` variants must remain one class (`R`). They must not be added as
separate classes such as `R_1` and `R_2`.

## Pre-training Checklist

- [ ] All new `r_1` and `r_2` images are stored under `data/raw_airwrite/R/`.
- [ ] All new `n` images are stored under `data/raw_airwrite/N/`.
- [ ] Images are grayscale PNG files with the expected 28x28 contract.
- [ ] The new images were collected with the same preprocessing and camera flow as production.
- [ ] The test split contains samples not used for training.
- [ ] Existing classes A-Z remain present.
- [ ] The old model artifact has been backed up or remains available for rollback.

## Rebuild and Training Commands

Run these commands from the v2 branch:

```powershell
python -m scripts.build_dataset_manifest
python -m scripts.preview_dataset_samples
python -m scripts.train_character_model
python -m scripts.evaluate_character_model
```

Record the exact branch, commit, training timestamp, random seed, and dataset
report path before replacing the runtime model.

## Dataset Snapshot After Manifest Rebuild

Record values from `data/manifests/dataset_report.json`.

| Field | Value |
| --- | --- |
| Dataset root |  |
| Scanned images |  |
| Valid unique images |  |
| Validation issues |  |
| Exact duplicates |  |
| Class count `R` |  |
| Class count `N` |  |
| Train / validation / test |  |
| Random seed |  |

### Recorded v2 Dataset Snapshot

The rebuilt manifest currently reports 3,993 scanned images, 3,986 valid
unique images, 5 exact duplicates and 2 preprocessing errors. The split is
2,790 train / 598 validation / 598 test. Class counts are `R=251` and
`N=200`; this is an intentionally changed test distribution compared with the
old baseline (`R=22`, `N=22` test support), so absolute accuracy comparisons
must be interpreted with care.

## V2 Results After Retraining

Record values from the new artifact directory after training and evaluation.

| Metric | Baseline | V2 | Change |
| --- | ---: | ---: | ---: |
| Test accuracy | 80.52% |  |  |
| Macro F1 | 80.62% |  |  |
| Top-3 accuracy | 92.17% |  |  |
| `R` precision | 80.95% |  |  |
| `R` recall | 77.27% |  |  |
| `R` F1 | 79.07% |  |  |
| `N` precision | 71.43% |  |  |
| `N` recall | 68.18% |  |  |
| `N` F1 | 69.77% |  |  |

### Recorded v2 Metrics

| Metric | Baseline | V2 | Change |
| --- | ---: | ---: | ---: |
| Test samples | 575 | 598 | +23 |
| Test accuracy | 80.52% | 80.77% | +0.25 pp |
| Macro precision |  — | 81.94% | — |
| Macro recall |  — | 80.98% | — |
| Macro F1 | 80.62% | 80.62% | ~0.00 pp |
| Top-3 accuracy | 92.17% | 92.31% | +0.14 pp |
| `R` precision | 80.95% | 86.11% | +5.16 pp |
| `R` recall | 77.27% | 81.58% | +4.31 pp |
| `R` F1 | 79.07% | 83.78% | +4.71 pp |
| `N` precision | 71.43% | 93.10% | +21.67 pp |
| `N` recall | 68.18% | 90.00% | +21.82 pp |
| `N` F1 | 69.77% | 91.53% | +21.76 pp |

## R/N Error Analysis

Fill this section from the new confusion matrix and error analysis:

- `R -> N`: baseline 2 test samples; v2: 1 test sample.
- `N -> R`: baseline count: 0 recorded in the baseline confusion row; v2: 0.
- `R` top-1 accuracy for style 1: 
- `R` top-1 accuracy for style 2: 
- `R` omitted from top-3: 
- `N` hard-negative accuracy: 
- Other classes with a material regression: 

## Acceptance Criteria

The v2 model is acceptable only if all of the following are reviewed:

- `R` style 1 and style 2 both improve or remain stable on an independent test set.
- `R -> N` errors decrease or do not materially worsen.
- `N -> R` errors do not increase enough to create a new user-facing problem.
- Overall accuracy and macro F1 do not regress materially.
- The model is tested with new camera samples after artifact export.
- The runtime preprocessing contract matches the trained model contract.

## Decision

Status: `PENDING`

Decision options: `KEEP_CURRENT`, `PROMOTE_V2`, or `REJECT_V2_AND_COLLECT_MORE_DATA`.

Decision notes:

_
