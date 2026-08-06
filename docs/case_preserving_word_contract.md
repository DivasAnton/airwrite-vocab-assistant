# Case-Preserving Word Contract

## Identity and Rendering

The EMNIST model predicts one canonical identity from `a` through `z`. It never predicts case.
`CaseLabelResolver` resolves that identity with an immutable `CaseSelection` before the Word Builder
receives it.

```text
identity=d + LOWERCASE -> rendered_character=d
identity=d + UPPERCASE -> rendered_character=D
```

The Word Builder validates this relationship and stores both values. It does not call `upper()` or
`lower()` to recreate prediction output.

## Supported Characters

`SupportedCharacterSet` contains three aligned tuples:

- 26 canonical identities `a-z`.
- 26 lowercase display characters `a-z`.
- 26 uppercase display characters `A-Z`.

The same tuple index always refers to the same letter identity. Spaces, digits, punctuation and
non-ASCII letters are outside the Sprint 11E contract.

## Word Forms

`current_word` and `ConfirmedWord.word` join `CharacterEntry.rendered_character` values exactly.
They distinguish `cat`, `Cat`, `CAT` and `AirWrite`.

`canonical_word` is derived with `casefold()` for future lookup. It never replaces or mutates the
original word.

## Pending Selection

All candidates in one pending prediction share the frozen case mode. Changing the global mode while
a decision is pending affects only later predictions; it cannot rewrite pending candidates.

## Shift Lifecycle

- Accepted commit using Shift: consume Shift after append succeeds.
- Pending candidate using Shift: keep Shift while pending; consume it after selection succeeds.
- Failed or empty prediction: keep Shift.
- Cancel Shift-based pending prediction: cancel Shift.
- Clear or start new word: cancel one-shot Shift and preserve locked mode.
- Backspace: remove one entry without restoring historical Shift state.

## Ownership

The Word Builder does not load a model, preprocess images, resolve case, call translation, validate a
dictionary or write persistent data. `app.main` composes the model mappings and coordinates case and
canvas side effects after domain actions succeed.
