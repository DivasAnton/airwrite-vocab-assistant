from app.inference.case_label_resolver import CaseLabelResolver, ResolvedCharacter
from app.inference.character_case_mode import CharacterCaseMode


class IdentityOnlyLabelResolver(CaseLabelResolver):
    """Render every recognized identity canonically as lowercase in v2."""

    def resolve(self, class_index: int, mode: CharacterCaseMode) -> ResolvedCharacter:
        resolved = super().resolve(class_index, CharacterCaseMode.LOWERCASE)
        return ResolvedCharacter(
            identity=resolved.identity,
            rendered_character=resolved.identity,
            class_index=resolved.class_index,
            case_mode=CharacterCaseMode.LOWERCASE,
        )
