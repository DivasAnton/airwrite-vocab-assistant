from dataclasses import dataclass

from app.inference.character_case_mode import CharacterCaseMode


@dataclass(frozen=True)
class SupportedCharacterSet:
    identities: tuple[str, ...]
    lowercase_characters: tuple[str, ...]
    uppercase_characters: tuple[str, ...]

    def __post_init__(self) -> None:
        mappings = {
            "identities": self.identities,
            "lowercase characters": self.lowercase_characters,
            "uppercase characters": self.uppercase_characters,
        }
        for name, values in mappings.items():
            if len(values) != 26:
                raise ValueError(f"{name} must contain exactly 26 characters")
            if any(len(value) != 1 for value in values):
                raise ValueError(f"every item in {name} must be one character")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")

        if self.identities != tuple("abcdefghijklmnopqrstuvwxyz"):
            raise ValueError("identities must follow the canonical a-z index contract")
        for identity, lowercase, uppercase in zip(
            self.identities,
            self.lowercase_characters,
            self.uppercase_characters,
            strict=True,
        ):
            if lowercase != identity:
                raise ValueError("lowercase characters must match the identity index contract")
            if uppercase.casefold() != identity or uppercase == lowercase:
                raise ValueError("uppercase characters must match the identity index contract")
        if len(set(self.all_characters)) != 52:
            raise ValueError("lowercase and uppercase character mappings must not overlap")

    @classmethod
    def english_letters(cls) -> "SupportedCharacterSet":
        identities = tuple("abcdefghijklmnopqrstuvwxyz")
        return cls(
            identities=identities,
            lowercase_characters=identities,
            uppercase_characters=tuple(character.upper() for character in identities),
        )

    @property
    def all_characters(self) -> tuple[str, ...]:
        return self.lowercase_characters + self.uppercase_characters

    @property
    def identity_count(self) -> int:
        return len(self.identities)

    def supports_entry(
        self,
        identity: str,
        rendered_character: str,
        case_mode: CharacterCaseMode | None,
    ) -> bool:
        if identity not in self.identities or case_mode is None:
            return False
        index = self.identities.index(identity)
        expected = (
            self.lowercase_characters[index]
            if case_mode is CharacterCaseMode.LOWERCASE
            else self.uppercase_characters[index]
        )
        return rendered_character == expected

    def identity_for(self, rendered_character: str) -> str:
        if rendered_character in self.lowercase_characters:
            return self.identities[self.lowercase_characters.index(rendered_character)]
        if rendered_character in self.uppercase_characters:
            return self.identities[self.uppercase_characters.index(rendered_character)]
        raise ValueError("rendered_character is not supported")

    def case_mode_for(self, rendered_character: str) -> CharacterCaseMode:
        if rendered_character in self.lowercase_characters:
            return CharacterCaseMode.LOWERCASE
        if rendered_character in self.uppercase_characters:
            return CharacterCaseMode.UPPERCASE
        raise ValueError("rendered_character is not supported")
