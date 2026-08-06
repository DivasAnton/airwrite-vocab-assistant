from app.word_builder.character_entry import CharacterEntry
from app.word_builder.character_source import CharacterSource
from app.word_builder.confirmed_word import ConfirmedWord
from app.word_builder.exceptions import PendingSelectionError
from app.word_builder.pending_character_selection import PendingCharacterSelection
from app.word_builder.supported_character_set import SupportedCharacterSet
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder_result import WordBuilderResult
from app.word_builder.word_builder_state import WordBuilderState


class WordBuilder:
    def __init__(
        self,
        supported_character_set: SupportedCharacterSet | None = None,
        max_length: int = 30,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")
        self.max_length = max_length
        self.supported_character_set = (
            supported_character_set or SupportedCharacterSet.english_letters()
        )
        self._entries: list[CharacterEntry] = []
        self._pending_selection: PendingCharacterSelection | None = None
        self._processed_prediction_ids: set[str] = set()
        self._last_confirmed_word: ConfirmedWord | None = None
        self._state = WordBuilderState.EMPTY

    @property
    def entries(self) -> tuple[CharacterEntry, ...]:
        return tuple(self._entries)

    @property
    def current_word(self) -> str:
        return "".join(entry.rendered_character for entry in self._entries)

    @property
    def canonical_word(self) -> str:
        return self.current_word.casefold()

    @property
    def pending_selection(self) -> PendingCharacterSelection | None:
        return self._pending_selection

    @property
    def last_confirmed_word(self) -> ConfirmedWord | None:
        return self._last_confirmed_word

    @property
    def state(self) -> WordBuilderState:
        return self._state

    @property
    def processed_prediction_ids(self) -> frozenset[str]:
        return frozenset(self._processed_prediction_ids)

    def has_prediction_id(self, prediction_id: str) -> bool:
        return prediction_id in self._processed_prediction_ids or (
            self._pending_selection is not None
            and self._pending_selection.prediction_id == prediction_id
        )

    def append_entry(self, entry: CharacterEntry) -> WordBuilderResult:
        if entry.prediction_id is not None and self.has_prediction_id(entry.prediction_id):
            return self.snapshot(WordAction.DUPLICATE_IGNORED, "Duplicate prediction ignored")
        if self._pending_selection is not None:
            return self.snapshot(WordAction.ERROR, "Resolve or cancel the pending character first")
        if self._state == WordBuilderState.CONFIRMED:
            return self.snapshot(WordAction.ERROR, "Start a new word before adding a character")
        if len(self._entries) >= self.max_length:
            return self.snapshot(WordAction.LIMIT_REACHED, "Word length limit reached")
        if not self.supported_character_set.supports_entry(
            entry.identity,
            entry.rendered_character,
            entry.case_mode,
        ):
            return self.snapshot(WordAction.ERROR, "Character entry is not supported")

        self._entries.append(entry)
        if entry.prediction_id is not None:
            self._processed_prediction_ids.add(entry.prediction_id)
        self._state = WordBuilderState.BUILDING
        return self.snapshot(
            WordAction.CHARACTER_APPENDED,
            f"Added: {entry.rendered_character}",
            committed_entry=entry,
        )

    def append_character(self, entry: CharacterEntry) -> WordBuilderResult:
        """Compatibility alias retained for Sprint 11 callers."""
        return self.append_entry(entry)

    def commit_entries_atomic(
        self,
        entries: tuple[CharacterEntry, ...],
        source_event_id: str,
    ) -> WordBuilderResult:
        if not source_event_id.strip() or not entries:
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Whole-word commit requires an event ID and at least one character",
            )
        if (
            self._entries
            or self._pending_selection is not None
            or self._state is not WordBuilderState.EMPTY
        ):
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Start whole-word input with an empty Word Builder",
            )
        prediction_ids = tuple(entry.prediction_id for entry in entries)
        if any(prediction_id is None for prediction_id in prediction_ids):
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Every whole-word character requires a prediction ID",
            )
        if (
            len(set(prediction_ids)) != len(prediction_ids)
            or source_event_id in self._processed_prediction_ids
        ):
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Duplicate whole-word prediction ignored",
            )
        if len(entries) > self.max_length:
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Whole word exceeds the Word Builder length limit",
            )
        if any(
            not self.supported_character_set.supports_entry(
                entry.identity,
                entry.rendered_character,
                entry.case_mode,
            )
            for entry in entries
        ):
            return self.snapshot(
                WordAction.WHOLE_WORD_COMMIT_REJECTED,
                "Whole word contains unsupported character data",
            )

        self._entries.extend(entries)
        self._processed_prediction_ids.add(source_event_id)
        self._processed_prediction_ids.update(
            prediction_id for prediction_id in prediction_ids if prediction_id is not None
        )
        self._state = WordBuilderState.BUILDING
        return self.snapshot(
            WordAction.WHOLE_WORD_COMMITTED,
            f"Whole word added: {self.current_word}",
            committed_entries=entries,
        )

    def create_pending_selection(
        self,
        selection: PendingCharacterSelection,
    ) -> WordBuilderResult:
        if self.has_prediction_id(selection.prediction_id):
            return self.snapshot(WordAction.DUPLICATE_IGNORED, "Duplicate prediction ignored")
        if self._pending_selection is not None:
            return self.snapshot(WordAction.ERROR, "Resolve or cancel the pending character first")
        if self._state == WordBuilderState.CONFIRMED:
            return self.snapshot(WordAction.ERROR, "Start a new word before adding a character")
        if len(self._entries) >= self.max_length:
            return self.snapshot(WordAction.LIMIT_REACHED, "Word length limit reached")
        if any(
            not self.supported_character_set.supports_entry(
                candidate.identity,
                candidate.rendered_character,
                candidate.case_mode,
            )
            for candidate in selection.candidates
        ):
            return self.snapshot(WordAction.ERROR, "Pending selection contains unsupported data")

        self._pending_selection = selection
        self._state = WordBuilderState.AWAITING_SELECTION
        return self.snapshot(
            WordAction.PENDING_SELECTION_CREATED,
            "Choose a pending character or cancel it",
        )

    def select_pending_candidate(self, rank: int) -> WordBuilderResult:
        if self._pending_selection is None:
            return self.snapshot(WordAction.ERROR, "No pending character to select")
        try:
            candidate = self._pending_selection.select_by_rank(rank)
        except PendingSelectionError as error:
            return self.snapshot(WordAction.ERROR, str(error))
        if len(self._entries) >= self.max_length:
            return self.snapshot(WordAction.LIMIT_REACHED, "Word length limit reached")

        pending_selection = self._pending_selection
        prediction_id = pending_selection.prediction_id
        case_mode = candidate.case_mode
        if case_mode is None:
            return self.snapshot(WordAction.ERROR, "Pending candidate has no case mode")
        entry = CharacterEntry(
            identity=candidate.identity,
            rendered_character=candidate.rendered_character,
            case_mode=case_mode,
            shift_was_active=pending_selection.case_selection.shift_was_active,
            confidence=candidate.confidence,
            source=CharacterSource.USER_SELECTED,
            prediction_id=prediction_id,
        )
        self._entries.append(entry)
        self._processed_prediction_ids.add(prediction_id)
        self._pending_selection = None
        self._state = WordBuilderState.BUILDING
        return self.snapshot(
            WordAction.PENDING_SELECTION_RESOLVED,
            f"Selected: {entry.rendered_character}",
            committed_entry=entry,
        )

    def cancel_pending_selection(self) -> WordBuilderResult:
        if self._pending_selection is None:
            return self.snapshot(WordAction.NONE, "No pending character to cancel")
        self._processed_prediction_ids.add(self._pending_selection.prediction_id)
        self._pending_selection = None
        self._state = WordBuilderState.BUILDING if self._entries else WordBuilderState.EMPTY
        return self.snapshot(
            WordAction.PENDING_SELECTION_CANCELLED,
            "Pending character cancelled",
        )

    def backspace(self) -> WordBuilderResult:
        if self._state == WordBuilderState.CONFIRMED:
            return self.snapshot(WordAction.ERROR, "Start a new word before editing")
        if not self._entries:
            return self.snapshot(WordAction.NONE, "No character to remove")

        removed = self._entries.pop()
        self._state = (
            WordBuilderState.AWAITING_SELECTION
            if self._pending_selection is not None
            else WordBuilderState.BUILDING
            if self._entries
            else WordBuilderState.EMPTY
        )
        return self.snapshot(WordAction.CHARACTER_REMOVED, f"Removed: {removed.character}")

    def clear(self) -> WordBuilderResult:
        self._reset()
        return self.snapshot(WordAction.WORD_CLEARED, "Word cleared")

    def confirm(self, timestamp_ms: int) -> WordBuilderResult:
        if self._pending_selection is not None:
            return self.snapshot(WordAction.ERROR, "Resolve or cancel the pending character first")
        if not self._entries:
            return self.snapshot(WordAction.ERROR, "Cannot confirm an empty word")
        if self._state == WordBuilderState.CONFIRMED:
            return self.snapshot(WordAction.NONE, "Word is already confirmed")

        confirmed_word = ConfirmedWord(
            word=self.current_word,
            entries=self.entries,
            confirmed_at_ms=timestamp_ms,
        )
        self._last_confirmed_word = confirmed_word
        self._state = WordBuilderState.CONFIRMED
        return self.snapshot(
            WordAction.WORD_CONFIRMED,
            f"Confirmed word: {confirmed_word.word}",
        )

    def start_new_word(self) -> WordBuilderResult:
        self._reset()
        return self.snapshot(WordAction.WORD_CLEARED, "Started a new word")

    def snapshot(
        self,
        action: WordAction = WordAction.NONE,
        message: str = "Word builder ready",
        committed_entry: CharacterEntry | None = None,
        committed_entries: tuple[CharacterEntry, ...] = (),
    ) -> WordBuilderResult:
        return WordBuilderResult(
            action=action,
            state=self._state,
            current_word=self.current_word,
            canonical_word=self.canonical_word,
            committed_entry=committed_entry,
            pending_selection=self._pending_selection,
            confirmed_word=self._last_confirmed_word,
            message=message,
            committed_entries=committed_entries,
        )

    def _reset(self) -> None:
        self._entries.clear()
        self._pending_selection = None
        self._processed_prediction_ids.clear()
        self._last_confirmed_word = None
        self._state = WordBuilderState.EMPTY
