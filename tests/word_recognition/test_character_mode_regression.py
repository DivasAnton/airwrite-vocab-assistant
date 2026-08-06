from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.input_mode_controller import InputModeController


def test_character_mode_remains_default_compatible_and_repeatable() -> None:
    controller = InputModeController(DrawingInputMode.CHARACTER)

    assert controller.current_mode is DrawingInputMode.CHARACTER
    assert not controller.set_character_mode()
    assert controller.set_word_mode()
    assert controller.set_character_mode()
    assert controller.current_mode is DrawingInputMode.CHARACTER
