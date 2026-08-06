# EMNIST Letters Identity Model Card

## Status

Trained. Experiment E02 with light augmentation is the selected Sprint 9E baseline.

## Intended Model

- Name: `airwrite_emnist_letter_identity`
- Version: `1.0.0`
- Task: 26-class letter-identity classification
- Input: normalized `float32`, shape `28x28x1`, light foreground on black background
- Output: softmax probabilities in canonical identity order `a-z`
- Case sensitive: false
- Case source: user-selected mode in Sprint 10E

## Architecture

Three convolution blocks with 32, 64 and 128 filters; Batch Normalization after each convolution;
Max Pooling after the first two blocks; Global Average Pooling; Dense 128; Dropout 0.30; Dense 26
softmax. E01 disables augmentation. E02 uses light rotation, translation and zoom during training.

## Evaluation Policy

E01 and E02 must be compared on the same persisted validation split. The official test split must
only be evaluated after selecting an experiment. AirWrite evaluation must use new, independent
uppercase-style and lowercase-style sessions and report style-specific accuracy.

## Results

- E02 validation accuracy: 94.89%.
- E02 validation Macro F1: 94.89%.
- Official EMNIST test accuracy: 94.37%.
- Official EMNIST test Macro F1: 94.38%.
- Official EMNIST test Top-3 accuracy: 99.50%.
- AirWrite custom accuracy: 64.12% across 1,600 unique samples.
- AirWrite custom Macro F1: 64.39%; Top-3 accuracy: 81.50%.
- Uppercase-style AirWrite accuracy: 62.17% across 1,306 samples.
- Lowercase-style AirWrite accuracy: 72.79% across 294 samples.

## Known Limitations

The model cannot infer case. The gap from 94.37% on official EMNIST to 64.12% on custom AirWrite
shows that the model is a useful baseline, not yet a production-ready recognizer. AirWrite recall is
especially weak for `r`, `n`, `b`, `d`, `e`, `x` and `u`. The uppercase and lowercase sample counts
are also imbalanced, so later product evaluation should collect additional independent sessions.
