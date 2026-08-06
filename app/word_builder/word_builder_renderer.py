from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.inference.case_input_state import CaseInputState
from app.inference.character_case_mode import CharacterCaseMode
from app.word_builder.word_builder_result import WordBuilderResult
from app.word_builder.word_builder_state import WordBuilderState


class WordBuilderRenderer:
    def __init__(
        self,
        max_length: int,
        enabled: bool = True,
        show_pending_candidates: bool = True,
        status_display_ms: int = 3000,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")
        if status_display_ms < 0:
            raise ValueError("status_display_ms must be greater than or equal to 0")
        self.max_length = max_length
        self.enabled = enabled
        self.show_pending_candidates = show_pending_candidates
        self.status_display_ms = status_display_ms

    def render(
        self,
        frame: NDArray[np.uint8],
        result: WordBuilderResult,
        status_started_ms: int,
        current_timestamp_ms: int,
        case_state: CaseInputState | None = None,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        if not self.enabled:
            return output

        show_status = (
            current_timestamp_ms >= status_started_ms
            and current_timestamp_ms - status_started_ms <= self.status_display_ms
        )
        lines = self.format_lines(result, case_state=case_state, show_status=show_status)
        line_height = 27
        panel_width = min(440, max(260, output.shape[1] - 20))
        panel_height = 18 + line_height * len(lines)
        left = max(8, output.shape[1] - panel_width - 10)
        top = 10
        right = min(output.shape[1] - 8, left + panel_width)
        bottom = min(output.shape[0] - 8, top + panel_height)
        overlay = output.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), (20, 20, 20), -1)
        output = cast(NDArray[np.uint8], cv2.addWeighted(overlay, 0.78, output, 0.22, 0.0))

        y = top + 25
        for line in lines:
            if y > bottom - 5:
                break
            cv2.putText(
                output,
                line,
                (left + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (80, 230, 230),
                2,
                cv2.LINE_AA,
            )
            y += line_height
        return output

    def format_lines(
        self,
        result: WordBuilderResult,
        *,
        case_state: CaseInputState | None = None,
        show_status: bool = True,
    ) -> list[str]:
        word_text = result.current_word or "_"
        lines = [
            f"Word: {word_text}",
            f"Length: {len(result.current_word)}/{self.max_length}",
        ]
        if case_state is not None:
            lines.append(f"Mode: {self._format_case_mode(case_state.locked_mode)}")
            if case_state.shift_next:
                lines.append(
                    f"Next character: {self._format_case_mode(case_state.get_effective_mode())}"
                )
        if result.state == WordBuilderState.CONFIRMED and result.confirmed_word is not None:
            lines.append(f"Confirmed: {result.confirmed_word.word}")
        elif result.pending_selection is not None:
            lines.append("Choose character: 1 / 2 / 3, X cancel")
            if self.show_pending_candidates:
                lines.extend(
                    f"{candidate.rank}. {candidate.rendered_character} {candidate.confidence:.1%}"
                    for candidate in result.pending_selection.candidates
                )
            lines.append(
                "Pending mode: "
                f"{self._format_case_mode(result.pending_selection.case_selection.mode)}"
            )
        else:
            lines.append("Enter confirm | B back | K clear | N new")
        if show_status:
            lines.append(result.message)
        return lines

    @staticmethod
    def _format_case_mode(mode: CharacterCaseMode) -> str:
        return "lowercase" if mode is CharacterCaseMode.LOWERCASE else "UPPERCASE"
