# Whole-Word Prediction Contract

Each ordered character segment is passed through the existing Sprint 8E `HandwritingPreprocessor`.
`CharacterBatchBuilder` creates one finite `float32` tensor with shape `(N, 28, 28, 1)` and values
in `[0, 1]`. `CharacterPredictor.predict_batch` invokes `model.predict` once for initial word
recognition and validates a `(N, 26)` softmax result.

Every position stores identity, rendered case, confidence, Top-3 candidates, prediction status,
segment ID, and bounding box. Uncertain positions and ambiguous segmentation block acceptance.
Candidate, case, split, and merge correction happen in the draft. Only a fully resolved draft can
be committed, and all entries are written to an empty Word Builder atomically.
