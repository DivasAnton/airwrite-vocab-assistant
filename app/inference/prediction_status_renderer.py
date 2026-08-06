from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.inference.case_input_state import CaseInputState
from app.inference.prediction_result import PredictionResult
from app.inference.prediction_status import PredictionStatus


class PredictionStatusRenderer:
    def __init__(
        self,
        show_top_k: bool = True,
        status_display_ms: int = 4000,
        show_model_version: bool = True,
        show_case_mode: bool = True,
        show_case_debug: bool = False,
    ) -> None:
        if status_display_ms < 0:
            raise ValueError("status_display_ms must be greater than or equal to 0")
        self.show_top_k = show_top_k
        self.status_display_ms = status_display_ms
        self.show_model_version = show_model_version
        self.show_case_mode = show_case_mode
        self.show_case_debug = show_case_debug

    def render(
        self,
        frame: NDArray[np.uint8],
        result: PredictionResult | None,
        display_started_ms: int,
        current_timestamp_ms: int,
        case_state: CaseInputState | None = None,
        compatibility_message: str | None = None,
        case_status_message: str | None = None,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        prediction_visible = self.is_visible(result, display_started_ms, current_timestamp_ms)
        lines = self.format_case_lines(case_state)
        if case_status_message:
            lines.append(case_status_message)
        if prediction_visible:
            assert result is not None
            lines.extend(self.format_lines(result))
            if compatibility_message:
                lines.append(compatibility_message)
        if not lines:
            return output
        color = (
            self._status_color(result.status) if prediction_visible and result else (220, 220, 220)
        )
        line_height = 28
        box_height = 18 + line_height * len(lines)
        height, width = output.shape[:2]
        top = max(0, height - box_height - 12)
        overlay = output.copy()
        cv2.rectangle(overlay, (8, top), (min(width - 8, 680), height - 8), (20, 20, 20), -1)
        output = cast(NDArray[np.uint8], cv2.addWeighted(overlay, 0.78, output, 0.22, 0.0))
        y = top + 26
        for line in lines:
            cv2.putText(
                output,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            y += line_height
        return output

    def format_case_lines(self, case_state: CaseInputState | None) -> list[str]:
        if not self.show_case_mode or case_state is None:
            return []
        locked_label = (
            "lowercase"
            if case_state.locked_mode.value == "LOWERCASE"
            else case_state.locked_mode.value
        )
        lines = [f"Mode: {locked_label}"]
        if case_state.shift_next:
            lines.append(f"Next character: {case_state.get_effective_mode().value}")
        return lines

    def is_visible(
        self,
        result: PredictionResult | None,
        display_started_ms: int,
        current_timestamp_ms: int,
    ) -> bool:
        return (
            result is not None
            and current_timestamp_ms >= display_started_ms
            and current_timestamp_ms - display_started_ms <= self.status_display_ms
        )

    def format_lines(self, result: PredictionResult) -> list[str]:
        if result.status == PredictionStatus.SKIPPED_EMPTY:
            lines = ["Nothing to predict"]
        elif result.status == PredictionStatus.FAILED:
            lines = [
                "Model unavailable"
                if result.message == "Model unavailable"
                else "Prediction failed"
            ]
        elif result.status == PredictionStatus.ACCEPTED:
            assert result.top_prediction is not None
            lines = [
                f"Prediction: {result.top_prediction.label}",
                f"Confidence: {result.top_prediction.confidence:.1%}",
            ]
        else:
            lines = ["Uncertain prediction"]
            candidates = result.candidates if self.show_top_k else result.candidates[:1]
            lines.extend(
                f"{candidate.rank}. {candidate.label} - {candidate.confidence:.1%}"
                for candidate in candidates
            )

        if self.show_model_version and result.model_version:
            lines.append(f"Model: v{result.model_version}")
        if self.show_case_debug and result.top_prediction is not None:
            lines.extend(
                [
                    f"Identity: {result.top_prediction.identity}",
                    f"Rendered: {result.top_prediction.rendered_character}",
                    "Case inferred by model: no",
                ]
            )
        return lines

    @staticmethod
    def _status_color(status: PredictionStatus) -> tuple[int, int, int]:
        if status == PredictionStatus.ACCEPTED:
            return (80, 220, 80)
        if status == PredictionStatus.UNCERTAIN:
            return (0, 210, 255)
        if status == PredictionStatus.FAILED:
            return (80, 80, 255)
        return (220, 220, 220)
