import numpy as np

from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.whole_word_renderer import WholeWordRenderer
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_writing_region import WordWritingRegion


def test_renderer_status_explains_single_done_whole_word_workflow() -> None:
    renderer = WholeWordRenderer()

    lines = renderer.status_lines(
        DrawingInputMode.ISOLATED_WORD,
        WordCasePolicy.CAPITALIZE_FIRST,
        None,
    )

    assert "Input: ISOLATED_WORD" in lines
    assert "Word case: CAPITALIZE_FIRST" in lines
    assert any("DONE once" in line for line in lines)


def test_character_mode_render_is_non_destructive() -> None:
    renderer = WholeWordRenderer()
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    output = renderer.render(
        frame,
        DrawingInputMode.CHARACTER,
        WordCasePolicy.LOWERCASE,
        WordWritingRegion(0, 0, 100, 50),
        None,
    )

    assert output is not frame
    assert np.array_equal(output, frame)
