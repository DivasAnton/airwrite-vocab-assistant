# EMNIST Letters Dataset Contract

## Source Of Truth

The source is the official NIST EMNIST Letters IDX distribution. TFDS may provide a download cache,
but its metadata is not trusted for class count or sample count.

## Observed Dataset

| Split | Samples | Samples per identity |
| --- | ---: | ---: |
| Official train | 124,800 | 4,800 |
| Official test | 20,800 | 800 |
| Derived train | 112,320 | 4,320 |
| Derived validation | 12,480 | 480 |

- Image storage: raw `uint8`, shape `28x28`.
- Raw label range: `1-26`.
- Mapped label range: `0-25`.
- Canonical identities: `a-z` in index order.
- Task: merged uppercase/lowercase letter identity, not case classification.

## Ownership Rules

- `IDXImageReader` validates and returns raw pixel orientation unchanged.
- `EMNISTLabelMapper` is the only component allowed to subtract one from labels.
- `EMNISTSourceAdapter` owns the single transpose required by the Sprint 8E visual audit.
- `ModelInputPreprocessor` owns the single `/255.0` normalization to `float32`.
- `EMNISTSplitBuilder` sees official train indices only; official test is never passed to it.
- Augmentation runs inside the model only while Keras is in training mode.

The machine-readable audit is stored locally at `data/external/emnist/audit.json`; dataset files are
ignored and must not be committed.
