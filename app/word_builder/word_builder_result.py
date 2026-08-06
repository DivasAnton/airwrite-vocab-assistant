from dataclasses import dataclass

from app.word_builder.character_entry import CharacterEntry
from app.word_builder.confirmed_word import ConfirmedWord
from app.word_builder.pending_character_selection import PendingCharacterSelection
from app.word_builder.word_action import WordAction
from app.word_builder.word_builder_state import WordBuilderState


@dataclass(frozen=True)
class WordBuilderResult:
    action: WordAction
    state: WordBuilderState
    current_word: str
    canonical_word: str
    committed_entry: CharacterEntry | None
    pending_selection: PendingCharacterSelection | None
    confirmed_word: ConfirmedWord | None
    message: str
    committed_entries: tuple[CharacterEntry, ...] = ()

    def __post_init__(self) -> None:
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for character in self.current_word
        ):
            raise ValueError("current_word must contain letters a-z or A-Z only")
        if self.canonical_word != self.current_word.casefold():
            raise ValueError("canonical_word must be the casefolded current_word")
        if not self.message.strip():
            raise ValueError("message must not be empty")
        if self.state == WordBuilderState.AWAITING_SELECTION and self.pending_selection is None:
            raise ValueError("AWAITING_SELECTION requires a pending selection")
        if self.state == WordBuilderState.CONFIRMED and self.confirmed_word is None:
            raise ValueError("CONFIRMED requires a confirmed word")
        if self.did_append_character and self.committed_entry is None:
            raise ValueError("append actions require the committed character entry")
        if self.did_commit_word and not self.committed_entries:
            raise ValueError("Whole-word commit requires committed entries")

    @property
    def did_append_character(self) -> bool:
        return self.action in {
            WordAction.CHARACTER_APPENDED,
            WordAction.PENDING_SELECTION_RESOLVED,
        }

    @property
    def did_commit_word(self) -> bool:
        return self.action is WordAction.WHOLE_WORD_COMMITTED
