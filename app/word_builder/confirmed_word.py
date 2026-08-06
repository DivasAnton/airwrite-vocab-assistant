from dataclasses import dataclass

from app.word_builder.character_entry import CharacterEntry


@dataclass(frozen=True)
class ConfirmedWord:
    word: str
    entries: tuple[CharacterEntry, ...]
    confirmed_at_ms: int

    def __post_init__(self) -> None:
        if not self.word:
            raise ValueError("confirmed word must not be empty")
        if not self.entries:
            raise ValueError("confirmed word must contain character entries")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for character in self.word
        ):
            raise ValueError("confirmed word must contain letters a-z or A-Z only")
        if self.word != "".join(entry.rendered_character for entry in self.entries):
            raise ValueError("confirmed word must match its character entries")
        if self.confirmed_at_ms < 0:
            raise ValueError("confirmed_at_ms must be greater than or equal to 0")

    @property
    def canonical_word(self) -> str:
        return self.word.casefold()
