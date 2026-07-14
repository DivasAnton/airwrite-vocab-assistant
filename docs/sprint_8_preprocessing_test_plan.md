# Sprint 8 Handwriting Image Preprocessing Test Plan

## Scope

Sprint 8 prepares saved AirCanvas drawings for future character-recognition models. The pipeline
accepts NumPy grayscale, BGR, or BGRA images and returns a centered `28x28` grayscale image plus a
`float32` normalized array in the `0.0-1.0` range.

## Automated Tests

| Test case | Expected result |
| --- | --- |
| Grayscale `uint8` input | Accepted by `ImageValidator` |
| BGR `uint8` input | Accepted and converted with OpenCV BGR ordering |
| BGRA `uint8` input | Accepted and converted with OpenCV BGRA ordering |
| `None`, empty, float, 1D, object, 5-channel input | Rejected with `InvalidImageError` |
| Black background with white stroke | Threshold keeps foreground |
| White background with black stroke and invert enabled | Stroke becomes foreground |
| Threshold `255` | Produces empty binary mask |
| Empty canvas | Raises `EmptyDrawingError` |
| Tiny noise below minimum foreground pixels | Raises `EmptyDrawingError` |
| Rectangle in center | Bounding box matches foreground |
| Stroke near left/right edge | Padding is clamped inside image bounds |
| Multiple strokes | Bounding box covers all strokes |
| Tall crop | Resized without horizontal stretching |
| Wide crop | Resized without vertical stretching |
| Small crop | Output shape remains fixed |
| Left/right translated drawing | Processed outputs are nearly identical |
| Saved Sprint 7 PNG | Can be read and processed through `process_file()` |
| Debug mode | Returns grayscale, binary, cropped, and processed stages |

## Manual Visual Checks

1. Draw a single character on the AirCanvas.
2. Trigger `DONE` or press the manual preprocessing key.
3. Confirm the `AirWrite Preprocessed` window shows a pixelated preview of the `28x28` output.
4. Test characters near the left, right, top, and bottom of the canvas.
5. Test a tall character such as `I`.
6. Test a wide character such as `T`.
7. Test a multi-stroke character such as `F`.
8. Confirm the preview is centered and not visibly stretched.
9. Enable `SAVE_PREPROCESS_DEBUG_IMAGES=true` only for debugging.
10. Confirm debug images are written under `data/preprocessed_debug/` and are ignored by Git.

## Non-Goals

- No model training.
- No prediction.
- No random runtime augmentation.
- No morphology or skeletonization by default.
- No preprocessing of every camera frame.
- No raw camera-frame storage.
