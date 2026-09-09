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
