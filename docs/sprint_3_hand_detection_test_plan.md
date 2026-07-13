# Sprint 3 Hand Detection Manual Test Plan

## Setup

- Download a MediaPipe Hand Landmarker `.task` model.
- Place it at `models/hand_landmarker.task`.
- Run the app from the project root with `python -m app.main`.

| Test case | Expected result |
| --- | --- |
| No hand visible | App does not crash and shows `Hands: 0` |
| One hand in frame | 21-point skeleton is drawn |
| Left hand | Handedness label is shown and noted for mirror behavior |
| Right hand | Handedness label is shown and noted for mirror behavior |
| Hand near camera | Landmarks stay attached or limitation is recorded |
| Hand far from camera | Detection limits are recorded |
| Hand leaves frame | Detection disappears safely |
| Hand re-enters frame | Detection resumes without restart |
| Tilted hand | Stability is recorded |
| Fast movement | App does not crash and jitter is observed |
| Low light | Detection rate is recorded |
| Complex background | Serious false positives are noted |
| Two hands | App detects up to `HAND_NUM_HANDS` |
| Run for 10 minutes | No crash or obvious resource leak |
| Wrong model path | Clear error is logged |
| Camera closes | Camera and landmarker are cleaned up |

## Performance Notes

| State | Average FPS |
| --- | --- |
| Sprint 2 camera only | To be measured locally |
| Sprint 3 no hand visible | To be measured locally |
| Sprint 3 one hand visible | To be measured locally |
| Sprint 3 fast movement | To be measured locally |

## Privacy Notes

- Raw camera frames are processed in memory only.
- No screenshots, videos, or raw frame folders are created.
- No camera data is uploaded.
