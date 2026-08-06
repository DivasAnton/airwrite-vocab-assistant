import cv2
import numpy as np
from numpy.typing import NDArray

from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.whole_word_draft import WholeWordDraft
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_writing_region import WordWritingRegion


class WholeWordRenderer:
    def __init__(
        self,
        *,
        show_roi: bool = True,
        show_segment_boxes: bool = True,
        show_prediction_confidence: bool = True,
    ) -> None:
        self.show_roi = show_roi
        self.show_segment_boxes = show_segment_boxes
        self.show_prediction_confidence = show_prediction_confidence

    def render(
        self,
        frame: NDArray[np.uint8],
        mode: DrawingInputMode,
        case_policy: WordCasePolicy,
        writing_region: WordWritingRegion | None,
        draft: WholeWordDraft | None,
        status_message: str | None = None,
        show_status: bool = True,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        if mode is not DrawingInputMode.ISOLATED_WORD:
            return output
        if self.show_roi and writing_region is not None:
            cv2.rectangle(
                output,
                (writing_region.x, writing_region.y),
                (writing_region.right - 1, writing_region.bottom - 1),
                (80, 220, 220),
                1,
            )
        if draft is not None and self.show_segment_boxes:
            for index, segment in enumerate(draft.segments):
                color = (40, 220, 80) if index != draft.selected_position else (0, 170, 255)
                box = segment.bounding_box
                cv2.rectangle(
                    output,
                    (box.x, box.y),
                    (box.right - 1, box.bottom - 1),
                    color,
                    2,
                )
        if not show_status:
            return output
        for line_index, line in enumerate(
            self.status_lines(mode, case_policy, draft, status_message)
        ):
            cv2.putText(
                output,
                line,
                (12, 24 + line_index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return output

    def status_lines(
        self,
        mode: DrawingInputMode,
        case_policy: WordCasePolicy,
        draft: WholeWordDraft | None,
        status_message: str | None = None,
    ) -> tuple[str, ...]:
        lines = [f"Input: {mode.value}", f"Word case: {case_policy.value}"]
        if draft is None:
            lines.append("Write isolated letters with visible spaces, then DONE once")
        else:
            selected = draft.selected_character
            lines.extend(
                (
                    f"Draft: {draft.current_word}",
                    f"Position: {draft.selected_position + 1}/{len(draft.characters)}",
                    "Candidates: "
                    + "  ".join(
                        f"{candidate.rank}:{candidate.rendered_character}"
                        for candidate in selected.candidates
                    ),
                    f"Review: {'READY' if draft.can_accept else 'REQUIRED'}",
                )
            )
            if self.show_prediction_confidence:
                lines.append(f"Confidence: {selected.confidence:.1%}")
        if status_message:
            lines.append(status_message)
        return tuple(lines)
