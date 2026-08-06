from app.word_recognition.drawing_input_mode import DrawingInputMode
from app.word_recognition.whole_word_prediction_result import WholeWordPredictionResult
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot
from app.word_recognition.word_recognition_strategy import WordRecognitionStrategy


class WholeWordRecognitionService:
    def __init__(self, isolated_word_strategy: WordRecognitionStrategy) -> None:
        self.isolated_word_strategy = isolated_word_strategy

    def recognize(
        self,
        snapshot: WordInputSnapshot,
        mode: DrawingInputMode,
        case_policy: WordCasePolicy,
    ) -> WholeWordPredictionResult:
        if mode is not DrawingInputMode.ISOLATED_WORD:
            raise ValueError("Whole-word recognition requires ISOLATED_WORD mode")
        return self.isolated_word_strategy.recognize(snapshot, case_policy)
