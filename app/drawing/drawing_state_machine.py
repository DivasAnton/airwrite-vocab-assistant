from app.drawing.drawing_state import DrawingState
from app.vision.gesture import Gesture


class DrawingStateMachine:
    def __init__(self, cooldown_ms: int = 500) -> None:
        if cooldown_ms < 0:
            raise ValueError(f"cooldown_ms must be greater than or equal to 0, got {cooldown_ms}")

        self.cooldown_ms = cooldown_ms
        self.state = DrawingState.IDLE
        self.last_transition_ms: int | None = None

    def update(self, gesture: Gesture, timestamp_ms: int) -> DrawingState:
        next_state = self._next_state(gesture)
        if next_state == self.state:
            return self.state
        if self._is_in_cooldown(timestamp_ms):
            return self.state

        self.state = next_state
        self.last_transition_ms = timestamp_ms
        return self.state

    def clear(self, timestamp_ms: int) -> DrawingState:
        self.state = DrawingState.CLEAR
        self.last_transition_ms = timestamp_ms
        return self.state

    def reset_to_ready(self, timestamp_ms: int) -> DrawingState:
        self.state = DrawingState.READY
        self.last_transition_ms = timestamp_ms
        return self.state

    def set_state(self, state: DrawingState, timestamp_ms: int) -> DrawingState:
        self.state = state
        self.last_transition_ms = timestamp_ms
        return self.state

    def _next_state(self, gesture: Gesture) -> DrawingState:
        if gesture == Gesture.NO_HAND:
            return DrawingState.IDLE

        if self.state == DrawingState.IDLE:
            return DrawingState.READY
        if self.state == DrawingState.READY:
            if gesture == Gesture.INDEX_ONLY:
                return DrawingState.WRITING
            return DrawingState.READY
        if self.state == DrawingState.WRITING:
            if gesture == Gesture.OPEN_PALM:
                return DrawingState.PAUSED
            if gesture == Gesture.FIST:
                return DrawingState.DONE
            return DrawingState.WRITING
        if self.state == DrawingState.PAUSED:
            if gesture == Gesture.INDEX_ONLY:
                return DrawingState.WRITING
            if gesture == Gesture.FIST:
                return DrawingState.DONE
            return DrawingState.PAUSED
        if self.state == DrawingState.DONE:
            if gesture == Gesture.OPEN_PALM:
                return DrawingState.READY
            return DrawingState.DONE
        if self.state == DrawingState.CLEAR:
            return DrawingState.READY

        return self.state

    def _is_in_cooldown(self, timestamp_ms: int) -> bool:
        if self.last_transition_ms is None:
            return False
        return timestamp_ms - self.last_transition_ms < self.cooldown_ms
