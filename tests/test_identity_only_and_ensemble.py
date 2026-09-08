import numpy as np

from app.inference.ensemble_character_predictor import EnsembleCharacterPredictor
from app.inference.identity_candidate import IdentityCandidate
from app.inference.identity_only_label_resolver import IdentityOnlyLabelResolver
from app.inference.character_case_mode import CharacterCaseMode
from app.inference.model_bundle import ModelBundle


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, images, verbose=0):
        return np.repeat(self.probabilities[np.newaxis, :], len(images), axis=0)


def bundle(probabilities, version):
    labels = tuple("abcdefghijklmnopqrstuvwxyz")
    return ModelBundle(
        model=FakeModel(probabilities), identity_labels=labels,
        lowercase_display_labels=labels,
        uppercase_display_labels=tuple(label.upper() for label in labels),
        model_version=version, task_type="airwrite_custom_identity", case_sensitive=False,
        case_source="identity_only", expected_input_shape=(28, 28, 1), num_classes=26,
        preprocessing_contract={},
    )


def test_identity_only_resolver_always_renders_lowercase() -> None:
    resolver = IdentityOnlyLabelResolver(
        tuple("abcdefghijklmnopqrstuvwxyz"), tuple("abcdefghijklmnopqrstuvwxyz"),
        tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )
    resolved = resolver.resolve(0, CharacterCaseMode.UPPERCASE)
    assert resolved.identity == resolved.rendered_character == "a"
    assert resolved.case_mode is CharacterCaseMode.LOWERCASE


def test_ensemble_keeps_custom_as_primary_signal() -> None:
    custom_scores = [0.0] * 26
    emnist_scores = [0.0] * 26
    custom_scores[1] = 0.70
    custom_scores[0] = 0.30
    emnist_scores[2] = 0.95
    emnist_scores[0] = 0.05
    predictor = EnsembleCharacterPredictor(
        custom=__import__("app.inference.character_predictor", fromlist=["CharacterPredictor"]).CharacterPredictor(bundle(custom_scores, "custom")),
        emnist=__import__("app.inference.character_predictor", fromlist=["CharacterPredictor"]).CharacterPredictor(bundle(emnist_scores, "emnist")),
    )
    result = predictor.predict(np.zeros((28, 28), dtype=np.float32))
    assert result.candidates[0].identity == "b"
