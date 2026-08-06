from collections.abc import Callable

from app.word_recognition.drawing_input_mode import DrawingInputMode


class InputModeController:
    def __init__(
        self,
        current_mode: DrawingInputMode,
        reset_temporary_state: Callable[[], None] | None = None,
    ) -> None:
        self.current_mode = current_mode
        self._reset_temporary_state = reset_temporary_state

    def set_character_mode(self) -> bool:
        return self._set_mode(DrawingInputMode.CHARACTER)

    def set_word_mode(self) -> bool:
        return self._set_mode(DrawingInputMode.ISOLATED_WORD)

    def _set_mode(self, mode: DrawingInputMode) -> bool:
        if mode is self.current_mode:
            return False
        if self._reset_temporary_state is not None:
            self._reset_temporary_state()
        self.current_mode = mode
        return True
