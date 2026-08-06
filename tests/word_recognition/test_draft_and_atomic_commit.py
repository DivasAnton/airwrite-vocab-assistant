from dataclasses import replace

import numpy as np

from app.inference.character_case_mode import CharacterCaseMode
from app.inference.prediction_candidate import PredictionCandidate
from app.inference.prediction_status import PredictionStatus
from app.preprocessing.bounding_box import BoundingBox
from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder import WordBuilder
from app.word_recognition.character_segment import CharacterSegment
from app.word_recognition.segmentation_status import SegmentationStatus
from app.word_recognition.whole_word_draft import WholeWordDraft, WholeWordDraftStatus
from app.word_recognition.whole_word_prediction_result import WholeWordPredictionResult
from app.word_recognition.whole_word_prediction_status import WholeWordPredictionStatus
from app.word_recognition.word_case_policy import WordCasePolicy
from app.word_recognition.word_prediction_character import WordPredictionCharacter


def candidate(identity: str, rank: int = 1, confidence: float = 0.9) -> PredictionCandidate:
    return PredictionCandidate(
        label=identity,
        identity=identity,
        rendered_character=identity,
        case_mode=CharacterCaseMode.LOWERCASE,
        class_index=ord(identity) - ord("a"),
        confidence=confidence,
        rank=rank,
    )


def prediction_character(
    position: int,
    identity: str,
    status: PredictionStatus = PredictionStatus.ACCEPTED,
) -> WordPredictionCharacter:
    alternatives = (candidate(identity), candidate("z", rank=2, confidence=0.05))
    return WordPredictionCharacter(
        position=position,
        identity=identity,
        rendered_character=identity,
        case_mode=CharacterCaseMode.LOWERCASE,
        confidence=alternatives[0].confidence,
        candidates=alternatives,
        status=status,
        source=CharacterSource.AUTO_ACCEPTED,
        segment_id=f"s{position}",
        bounding_box=BoundingBox(position * 30, 0, 20, 30),
    )


def segment(position: int) -> CharacterSegment:
    return CharacterSegment(
        segment_id=f"s{position}",
        position=position,
        bounding_box=BoundingBox(position * 30, 0, 20, 30),
        source_stroke_ids=(),
        grayscale_image=np.full((30, 20), 255, dtype=np.uint8),
        segmentation_confidence=0.9,
    )


def result(
    characters: tuple[WordPredictionCharacter, ...],
    segmentation_status: SegmentationStatus = SegmentationStatus.SUCCESS,
) -> WholeWordPredictionResult:
    return WholeWordPredictionResult(
        prediction_id="word-event",
        status=WholeWordPredictionStatus.READY,
        characters=characters,
        predicted_word="".join(item.rendered_character for item in characters),
        segmentation_time_ms=1.0,
        preprocessing_time_ms=1.0,
        inference_time_ms=1.0,
        total_time_ms=3.0,
        segmentation_status=segmentation_status,
        segments=tuple(segment(index) for index in range(len(characters))),
        message="ready",
    )


def test_draft_requires_uncertain_position_to_be_resolved() -> None:
    draft = WholeWordDraft.from_prediction(
        result(
            (
                prediction_character(0, "c"),
                prediction_character(1, "a", PredictionStatus.UNCERTAIN),
                prediction_character(2, "t"),
            )
        ),
        WordCasePolicy.LOWERCASE,
    )

    assert draft.current_word == "cat"
    assert draft.unresolved_positions == (1,)
    draft.move_next()
    draft.select_candidate(1)
    assert draft.can_accept


def test_draft_navigation_case_toggle_and_cancel() -> None:
    draft = WholeWordDraft.from_prediction(
        result((prediction_character(0, "c"), prediction_character(1, "a"))),
        WordCasePolicy.LOWERCASE,
    )

    draft.move_next()
    draft.toggle_selected_case()
    assert draft.current_word == "cA"
    assert draft.selected_character.case_mode is CharacterCaseMode.UPPERCASE
    draft.cancel()
    assert draft.status is WholeWordDraftStatus.CANCELLED


def entry(character: str, event: str) -> CharacterEntry:
    return CharacterEntry(
        identity=character.lower(),
        rendered_character=character,
        case_mode=(
            CharacterCaseMode.UPPERCASE if character.isupper() else CharacterCaseMode.LOWERCASE
        ),
        shift_was_active=False,
        confidence=0.9,
        source=CharacterSource.AUTO_ACCEPTED,
        prediction_id=event,
    )


def test_whole_word_commit_is_atomic_and_requires_empty_builder() -> None:
    builder = WordBuilder(max_length=4)
    committed = builder.commit_entries_atomic(
        (entry("C", "word:0"), entry("a", "word:1"), entry("t", "word:2")),
        "word",
    )

    assert committed.action is WordAction.WHOLE_WORD_COMMITTED
    assert committed.current_word == "Cat"
    assert committed.committed_entries == builder.entries
    rejected = builder.commit_entries_atomic((entry("s", "other:0"),), "other")
    assert rejected.action is WordAction.WHOLE_WORD_COMMIT_REJECTED
    assert builder.current_word == "Cat"


def test_failed_atomic_validation_leaves_builder_unchanged() -> None:
    builder = WordBuilder(max_length=2)
    rejected = builder.commit_entries_atomic(
        (entry("c", "word:0"), entry("a", "word:1"), entry("t", "word:2")),
        "word",
    )

    assert rejected.action is WordAction.WHOLE_WORD_COMMIT_REJECTED
    assert builder.current_word == ""
    assert builder.entries == ()


def test_replacing_segment_range_renumbers_positions() -> None:
    draft = WholeWordDraft.from_prediction(
        result((prediction_character(0, "a"), prediction_character(1, "b"))),
        WordCasePolicy.LOWERCASE,
    )
    child_one = replace(segment(0), segment_id="left")
    child_two = replace(segment(0), segment_id="right")
    char_one = replace(prediction_character(0, "c"), segment_id="left")
    char_two = replace(prediction_character(1, "d"), segment_id="right")

    draft.replace_range(0, 1, (child_one, child_two), (char_one, char_two))

    assert [item.position for item in draft.characters] == [0, 1, 2]
    assert draft.current_word == "cdb"
