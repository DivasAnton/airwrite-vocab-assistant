from dataclasses import dataclass

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.drawing.stroke_manager import StrokeManager
from app.vision.finger_tracking_result import FingerTrackingResult
from app.vision.gesture import Gesture


@dataclass(frozen=True)
class DrawingControllerResult:
    state: DrawingState
    did_draw: bool
    did_clear: bool


class DrawingController:
    def __init__(
        self,
        state_machine: DrawingStateMachine,
        stroke_manager: StrokeManager,
        canvas: AirCanvas,
    ) -> None:
        self.state_machine = state_machine
        self.stroke_manager = stroke_manager
        self.canvas = canvas
        self.previous_state = state_machine.state

    def update(
        self,
        gesture: Gesture,
        finger_result: FingerTrackingResult,
    ) -> DrawingControllerResult:
        state = self.state_machine.update(gesture, finger_result.timestamp_ms)
        self._handle_state_transition(state)

        if state != DrawingState.WRITING:
            return DrawingControllerResult(state=state, did_draw=False, did_clear=False)

        segment = self.stroke_manager.update(finger_result.smoothed_point)
        if segment is None:
            return DrawingControllerResult(state=state, did_draw=False, did_clear=False)

        self.canvas.draw_line(segment.start, segment.end)
        return DrawingControllerResult(state=state, did_draw=True, did_clear=False)

    def clear(self, timestamp_ms: int) -> DrawingControllerResult:
        self.canvas.clear()
        self.stroke_manager.reset()
        self.state_machine.clear(timestamp_ms)
        state = self.state_machine.reset_to_ready(timestamp_ms)
        self.previous_state = state
        return DrawingControllerResult(state=state, did_draw=False, did_clear=True)

    def set_state(self, state: DrawingState, timestamp_ms: int) -> DrawingControllerResult:
        new_state = self.state_machine.set_state(state, timestamp_ms)
        if new_state != DrawingState.WRITING:
            self.stroke_manager.reset()
        self.previous_state = new_state
        return DrawingControllerResult(state=new_state, did_draw=False, did_clear=False)

    def _handle_state_transition(self, state: DrawingState) -> None:
        if self.previous_state == state:
            return
        if self.previous_state == DrawingState.WRITING or state in {
            DrawingState.WRITING,
            DrawingState.PAUSED,
            DrawingState.DONE,
            DrawingState.IDLE,
        }:
            self.stroke_manager.reset()
        self.previous_state = state
