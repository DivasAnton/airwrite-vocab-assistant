# Case-Controlled Inference Contract

## Pipeline

```text
AirWrite canvas snapshot
-> Sprint 8E AirWrite preprocessing
-> float32 tensor 28x28x1 in range 0..1
-> E02 identity model
-> IdentityCandidate a-z
-> confidence and margin policy
-> immutable CaseSelection
-> CaseLabelResolver
-> rendered PredictionCandidate a-z or A-Z
```

The model never infers case. `case_was_inferred` must remain `false`.

## Bundle Contract

The runtime loads exactly these configured artifacts once:

- `model.keras`
- `identity_labels.json`
- `lowercase_display_labels.json`
- `uppercase_display_labels.json`
- `model_metadata.json`
- `preprocessing_config.json`

Required metadata is `task_type=letter_identity_classification`, `case_sensitive=false`,
`case_source=user_selected_mode`, input `28x28x1`, and 26 outputs. Mapping index order must be
canonical `a-z`, `a-z`, and `A-Z`. Missing or incompatible artifacts disable prediction; the loader
does not search for or fall back to the legacy uppercase model.

## Case State

`CaseInputState.locked_mode` is lowercase or uppercase. `shift_next` temporarily selects the opposite
effective mode. Changing locked mode cancels Shift. `CaseSelection` copies the effective mode and
`shift_was_active` at the prediction event and is immutable.

Prediction, low confidence, and switching global mode do not consume or rewrite Shift. The method
`consume_shift_after_commit()` exists for Sprint 11E to call only after a character is appended.

## Prediction Semantics

`IdentityCandidate` contains identity, class index, confidence, and rank. `PredictionCandidate`
adds `rendered_character` and `case_mode`; its compatibility field `label` always equals
`rendered_character`. Case resolution cannot alter probability, rank, confidence margin, or model
call count.

## Legacy Word Builder Guard

Uppercase candidates may enter the current uppercase-only Word Builder. Lowercase candidates remain
visible in the prediction overlay but are not converted and are not passed into the Word Builder.
The user sees that lowercase word building requires Sprint 11E.

## Keyboard Contract

| Action | Default key |
| --- | --- |
| Set lowercase | `L` |
| Set uppercase | `U` |
| Toggle Shift-next | `Y` |
| Cancel Shift-next | `Z` |
| Manual prediction | `I` |

There is no AUTO case mode. Setting `ENABLE_AUTO_CASE_MODE=true` fails configuration validation.
