# Whole-Word Input Contract

- Mode: `ISOLATED_WORD`.
- Scope: one English word per canvas, 2-12 letters by default.
- Writing: left to right, isolated letters, visible horizontal gaps.
- Completion: one FIST/DONE gesture or `D` key after the complete word.
- ROI: configured by `WORD_ROI_*_RATIO`; points outside the ROI do not draw or record.
- Snapshot: clean AirCanvas pixels, completed stroke trajectories, ROI, event ID, and timestamp.
- Model: unchanged 26-class case-neutral E02 identity model.
- Case: selected by word policy; it is never inferred by the model.
- DONE creates a draft. `A` accepts the resolved draft; Enter confirms the resulting Word Builder.

Connected/cursive letters, phrases, spaces, punctuation, and digits violate this contract.
