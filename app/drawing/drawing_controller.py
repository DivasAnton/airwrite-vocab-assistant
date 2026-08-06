from dataclasses import dataclass

from app.drawing.air_canvas import AirCanvas
from app.drawing.drawing_state import DrawingState
from app.drawing.drawing_state_machine import DrawingStateMachine
from app.drawing.stroke_manager import StrokeManager
from app.vision.finger_tracking_result import FingerTrackingResult
from app.vision.gesture import Gesture
from app.word_recognition.stroke_point import StrokePoint
from app.word_recognition.stroke_recorder import StrokeRecorder
from app.word_recognition.word_writing_region import WordWritingRegion


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
        stroke_recorder: StrokeRecorder | None = None,
        writing_region: WordWritingRegion | None = None,
    ) -> None:
        self.state_machine = state_machine
        self.stroke_manager = stroke_manager
        self.canvas = canvas
        self.stroke_recorder = stroke_recorder
        self.writing_region = writing_region
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

        point = finger_result.smoothed_point
        if point is None or (
            self.writing_region is not None and not self.writing_region.contains(point)
        ):
            self.stroke_manager.update(None)
            if self.stroke_recorder is not None:
                self.stroke_recorder.end_stroke()
            return DrawingControllerResult(state=state, did_draw=False, did_clear=False)

        if self.stroke_recorder is not None:
            stroke_point = StrokePoint(point[0], point[1], finger_result.timestamp_ms)
            try:
                if self.stroke_recorder.is_recording:
                    self.stroke_recorder.append_point(stroke_point)
                else:
                    self.stroke_recorder.start_stroke(stroke_point)
            except ValueError:
                self.stroke_recorder.end_stroke()

        segment = self.stroke_manager.update(point)
        if segment is None:
            return DrawingControllerResult(state=state, did_draw=False, did_clear=False)

        self.canvas.draw_line(segment.start, segment.end)
        return DrawingControllerResult(state=state, did_draw=True, did_clear=False)

    def clear(self, timestamp_ms: int) -> DrawingControllerResult:
        self.canvas.clear()
        self.stroke_manager.reset()
        if self.stroke_recorder is not None:
            self.stroke_recorder.reset()
        self.state_machine.clear(timestamp_ms)
        state = self.state_machine.reset_to_ready(timestamp_ms)
        self.previous_state = state
        return DrawingControllerResult(state=state, did_draw=False, did_clear=True)

    def set_state(self, state: DrawingState, timestamp_ms: int) -> DrawingControllerResult:
        new_state = self.state_machine.set_state(state, timestamp_ms)
        if new_state != DrawingState.WRITING:
            self.stroke_manager.reset()
            if self.stroke_recorder is not None:
                self.stroke_recorder.end_stroke()
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
        if self.previous_state == DrawingState.WRITING and self.stroke_recorder is not None:
            self.stroke_recorder.end_stroke()
        self.previous_state = state

    def configure_word_input(
        self,
        stroke_recorder: StrokeRecorder | None,
        writing_region: WordWritingRegion | None,
    ) -> None:
        if self.stroke_recorder is stroke_recorder and self.writing_region == writing_region:
            return
        if self.stroke_recorder is not None and self.stroke_recorder is not stroke_recorder:
            self.stroke_recorder.end_stroke()
        self.stroke_manager.reset()
        self.stroke_recorder = stroke_recorder
        self.writing_region = writing_region
