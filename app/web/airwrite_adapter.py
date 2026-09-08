"""Web boundary for the existing AirWrite inference services.

The adapter deliberately does not load a model. The desktop runtime owns model
lifecycle; a web host injects the already-configured service at startup.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.inference.case_selection import CaseSelection
from app.inference.character_recognition_service import CharacterRecognitionService


class AirWriteRecognitionAdapter:
    def __init__(self, character_service: CharacterRecognitionService) -> None:
        self.character_service = character_service

    def recognize_character(
        self,
        canvas: NDArray[np.uint8],
        *,
        case_selection: CaseSelection,
        prediction_id: str | None = None,
    ) -> dict[str, Any]:
        result = self.character_service.recognize(
            canvas,
            case_selection=case_selection,
            prediction_id=prediction_id,
        )
        return {
            "prediction_id": result.prediction_id,
            "status": result.status.value,
            "prediction": result.top_prediction.rendered_character if result.top_prediction else None,
            "confidence": result.top_prediction.confidence if result.top_prediction else None,
            "margin": result.confidence_margin,
            "candidates": [
                {"label": item.rendered_character, "confidence": item.confidence, "rank": item.rank}
                for item in result.candidates
            ],
            "message": result.message,
        }


def ensure_canvas_uint8(canvas: np.ndarray) -> NDArray[np.uint8]:
    """Validate the web boundary without accepting arbitrary object arrays."""
    if canvas.ndim not in (2, 3) or canvas.size == 0:
        raise ValueError("canvas must be a non-empty 2D or 3D image")
    if canvas.dtype != np.uint8:
        raise ValueError("canvas must use uint8 pixels")
    return canvas

