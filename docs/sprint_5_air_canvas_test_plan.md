# Sprint 5 Air Drawing Canvas Manual Test Plan

## Scope

- Draw continuous line segments from the smoothed index fingertip point.
- Keep a persistent canvas independent from the camera frame.
- Overlay the canvas onto the camera preview.
- Clear with the configured keyboard key.
- Do not add gestures, save images, preprocess handwriting, or run prediction.

| Test case | Expected result |
| --- | --- |
| No hand visible | Canvas does not change |
| Hand appears for the first time | No long line is drawn |
| Index finger moves slowly | Stroke is continuous |
| Index finger moves quickly | Stroke follows hand or skips large jumps |
| Hand disappears | Stroke stops |
| Hand reappears elsewhere | Stroke does not connect to old point |
| Press `C` | Canvas clears |
| Clear then write again | New stroke does not connect to the pre-clear point |
| Canvas window | Shows only the independent drawing canvas |
| Overlay window | Stroke aligns with the camera preview |
| Run for 10 minutes | No crash or obvious leak |
| Finger near frame edge | Stroke does not exceed image array bounds |
| Resolution changes | Canvas resets safely |

## Character Smoke Test

Draw these simple characters to observe continuity, jitter, and thickness:

```text
I
L
C
V
O
```

Draw these asymmetric characters to verify direction and mirror behavior:

```text
F
R
K
```

## Privacy Notes

- Canvas and camera frames remain in memory only.
- Sprint 5 does not call `cv2.imwrite`.
- Sprint 5 does not upload camera or canvas data.
