from collections.abc import Mapping

from app.inference.character_case_mode import CharacterCaseMode
from app.word_recognition.word_case_policy import WordCasePolicy


class WordCaseResolver:
    def resolve_modes(
        self,
        count: int,
        policy: WordCasePolicy,
        overrides: Mapping[int, CharacterCaseMode] | None = None,
    ) -> tuple[CharacterCaseMode, ...]:
        if count <= 0:
            raise ValueError("Word character count must be positive")
        if policy is WordCasePolicy.UPPERCASE:
            modes = [CharacterCaseMode.UPPERCASE] * count
        elif policy is WordCasePolicy.CAPITALIZE_FIRST:
            modes = [CharacterCaseMode.LOWERCASE] * count
            modes[0] = CharacterCaseMode.UPPERCASE
        else:
            modes = [CharacterCaseMode.LOWERCASE] * count
        for position, mode in (overrides or {}).items():
            if not 0 <= position < count:
                raise ValueError("Word case override position is out of range")
            modes[position] = mode
        return tuple(modes)

    def resolve_word(
        self,
        identities: tuple[str, ...],
        policy: WordCasePolicy,
        overrides: Mapping[int, CharacterCaseMode] | None = None,
    ) -> tuple[str, ...]:
        modes = self.resolve_modes(len(identities), policy, overrides)
        return tuple(
            identity if mode is CharacterCaseMode.LOWERCASE else identity.upper()
            for identity, mode in zip(identities, modes, strict=True)
        )
