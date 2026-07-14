from app.vision.gesture import Gesture


class GestureStabilizer:
    def __init__(self, stable_frames: int = 5, lost_hand_frames: int = 10) -> None:
        if stable_frames <= 0:
            raise ValueError(f"stable_frames must be greater than 0, got {stable_frames}")
        if lost_hand_frames <= 0:
            raise ValueError(f"lost_hand_frames must be greater than 0, got {lost_hand_frames}")

        self.stable_frames = stable_frames
        self.lost_hand_frames = lost_hand_frames
        self.candidate_gesture = Gesture.NO_HAND
        self.candidate_frame_count = 0
        self.stable_gesture = Gesture.NO_HAND

    def update(self, raw_gesture: Gesture) -> Gesture:
        if raw_gesture == self.candidate_gesture:
            self.candidate_frame_count += 1
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_frame_count = 1

        required_frames = (
            self.lost_hand_frames if raw_gesture == Gesture.NO_HAND else self.stable_frames
        )
        if self.candidate_frame_count >= required_frames:
            self.stable_gesture = self.candidate_gesture

        return self.stable_gesture

    def reset(self) -> None:
        self.candidate_gesture = Gesture.NO_HAND
        self.candidate_frame_count = 0
        self.stable_gesture = Gesture.NO_HAND
