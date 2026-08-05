# AirWrite Character Recognizer Model Card

## Model Details

- Name: `airwrite_character_recognizer`
- Version: `0.1.0`
- Framework: TensorFlow/Keras
- Status: Trained
- Input: one `28x28x1` float32 image in the `0.0-1.0` range
- Output: 26 softmax scores ordered from A through Z

## Intended Use

The model recognizes one uppercase English character drawn through the AirWrite canvas. It is
intended for the local AirWrite Vocabulary Assistant prototype and for later Sprint 10 inference.

It is not intended for full words, lowercase-only recognition, digits, signatures, identity
verification, document OCR, or safety-critical decisions.

## Dataset

- Source: custom AirWrite drawings produced by the Sprint 8 preprocessing workflow
- Scanned images: 1,312
- Valid unique images: 1,306
- Classes: uppercase A-Z
- Train: 914
- Validation: 196
- Test: 196
- Exact duplicates excluded from manifests: 6
- Split seed: 42
- Split strategy: deterministic stratified 70/15/15 split

The current data appears to come from a limited collection setup. Generalization to other writers,
cameras, drawing speeds, hand motion styles, and environments has not yet been established.

## Preprocessing

The custom dataset images are already Sprint 8 outputs, so training does not crop and resize them a
second time. The loader verifies grayscale `28x28` PNG input and performs only `uint8` to `float32`
normalization. Runtime inference must use the same Sprint 8 contract before calling the model.

## Architecture

- Light rotation (`0.01`), translation (`0.03`), and zoom (`0.03`) augmentation during training only
- Three convolution blocks with 32, 64, and 128 filters
- Max pooling after the 32-filter and 64-filter convolution blocks
- No batch normalization
- Global average pooling
- Dense 64 with ReLU
- Dropout 0.2
- Dense 26 with softmax
- No horizontal or vertical flip

## Evaluation

The current `v0.1.0` checkpoint completed all 50 configured epochs. The best validation loss was
recorded at epoch 50.

- Validation accuracy: 81.63%
- Test accuracy: 79.08%
- Macro F1: 77.38%
- Top-3 accuracy: 90.31%
- Test samples: 196

Best epoch 50 indicates that the model may continue improving with a longer training schedule.
However, this fixed test set has already been inspected and must not be used for further architecture
or hyperparameter tuning. The current model is retained as the Sprint 10 baseline. The next rigorous
evaluation should use a new AirWrite collection session that has not been viewed or used during
Sprint 9 development.

## Known Limitations

- Recognizes a single uppercase English character only.
- Similar shapes such as I/L, C/G, O/Q, F/P, and U/V may be confused.
- Missing, interrupted, or unusually thick strokes may reduce accuracy.
- The dataset has not been broadly evaluated across multiple writers.
- Softmax always returns a top class; Sprint 10 still needs low-confidence handling.
- Test data must not be reused for architecture or hyperparameter tuning.

## Sprint 10 Handoff

The following classes have low test recall and require focused monitoring during live prediction:

- D: 28.6%
- F: 37.5%
- G: 37.5%
- H: 37.5%
- A: 42.9%
- B: 50.0%

Sprint 10 should retain top-3 probabilities, provide a low-confidence state, and record mistakes for
these classes without tuning against the current test set. Additional evaluation data should come
from a newly collected AirWrite session.

## Privacy

The model uses canvas images rather than raw camera video. Dataset files and model binaries are kept
local by default and are excluded from Git according to repository policy.
