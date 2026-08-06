from typing import Protocol

from app.word_recognition.whole_word_prediction_result import WholeWordPredictionResult
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_input_snapshot import WordInputSnapshot


class WordRecognitionStrategy(Protocol):
    def recognize(
        self,
        snapshot: WordInputSnapshot,
        case_policy: WordCasePolicy,
    ) -> WholeWordPredictionResult: ...
