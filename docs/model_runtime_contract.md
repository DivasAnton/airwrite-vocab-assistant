# Character Model Runtime Contract

## Frozen Bundle

Sprint 10 uses model version `0.1.0` and loads these configured artifacts once at application
startup:

- `artifacts/models/character_recognizer.keras`
- `artifacts/labels/labels.json`
- `artifacts/metadata/model_metadata.json`
- `artifacts/metadata/preprocessing_config.json`

The loader uses `keras.saving.load_model(..., compile=False, safe_mode=True)`. It never searches for
another model, trains, evaluates, or overwrites the artifact. The model SHA-256 in metadata is
validated before use.

## Labels And Shapes

- Labels are exactly uppercase `A` through `Z` in that order. Runtime never sorts or repairs them.
- Model input is `(None, 28, 28, 1)`.
- Model output is `(None, 26)` and already contains softmax probabilities.
- One normalized image becomes a batch with shape `(1, 28, 28, 1)`.
- Runtime does not apply a second normalization or a second softmax.

## Preprocessing

The runtime configuration must exactly match the frozen preprocessing artifact:

| Field | Value |
| --- | --- |
| output width / height | `28 / 28` |
| channels | `1` |
| background / foreground | `black / light` |
| normalized range | `0.0-1.0` |
| content width / height | `20 / 20` |
| binary threshold | `20` |
| crop padding / minimum foreground pixels | `8 / 10` |
| invert input / center of mass | `false / false` |

The Sprint 8 preprocessor returns a finite `float32` two-dimensional array. Any artifact, shape,
dtype, range, or preprocessing mismatch disables prediction instead of being silently repaired.

## Runtime Policy

The default policy accepts a prediction only when Top-1 confidence is at least `0.60` and the
Top-1 minus Top-2 margin is at least `0.15`. Equality passes. All classes use the same thresholds;
there are no special rules for A, B, D, F, G, or H.

`DONE` creates one in-memory canvas snapshot. The same array is sent to local PNG storage and the
recognition service. Camera frames are never model inputs. Manual prediction uses the current
canvas without saving, clearing, or changing drawing state.

## Failure Behavior

Missing or incompatible artifacts disable prediction while camera, drawing, and saving remain
available. Empty canvases return `SKIPPED_EMPTY`; low-confidence results return `UNCERTAIN`; model
or preprocessing failures return `FAILED`. Technical details go to logs, not the camera overlay.
# Continuous-Word CRNN Runtime Contract

The Sprint 11C-V3 model is independent of the character identity bundle described below.

- Artifact: Keras model at `CONTINUOUS_CRNN_MODEL_PATH`, loaded once with `compile=False`.
- Input: finite `float32`, shape `(1, 32, 128, 1)`, values in `[0, 1]`.
- Foreground: light strokes on a black background.
- Geometry: full foreground crop, aspect-preserving resize, vertical centering, right padding.
- Output: finite logits/probabilities shaped `(1, T, 27)`.
- Classes: canonical `a-z` at indices `0-25`; CTC blank at index `26`.
- Decode: prefix beam search with configured Top-K and beam width.
- Scope: one ASCII English word, 2-12 characters by default.
- Case: selected by `WordCasePolicy`; never inferred by the CRNN.
- Threading: inference runs on the controller's single worker; the camera loop polls completion.
- Failure isolation: an unavailable artifact disables continuous mode only.

The associated `metadata.json` must record model name/version, input shape, alphabet, blank index,
training/validation counts, CER, exact match, and whether the run was a smoke test. A smoke artifact
must not be used as the default runtime model.
