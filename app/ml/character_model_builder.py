import importlib
from dataclasses import dataclass
from typing import Any

from app.ml.exceptions import TrainingDependencyError


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    rotation_factor: float = 0.03
    translation_factor: float = 0.08
    zoom_factor: float = 0.08

    def validate(self) -> None:
        factors = (self.rotation_factor, self.translation_factor, self.zoom_factor)
        if any(factor < 0.0 or factor >= 1.0 for factor in factors):
            raise ValueError("Augmentation factors must be greater than or equal to 0 and below 1")


class CharacterModelBuilder:
    def __init__(
        self,
        augmentation_config: AugmentationConfig | None = None,
        learning_rate: float = 0.001,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.augmentation_config.validate()
        self.learning_rate = learning_rate

    def build(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 26,
    ) -> Any:
        if len(input_shape) != 3 or any(dimension <= 0 for dimension in input_shape):
            raise ValueError(
                f"input_shape must contain three positive dimensions, got {input_shape}"
            )
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        tensorflow = self._require_tensorflow()
        keras = tensorflow.keras
        inputs = keras.Input(shape=input_shape, name="character_image")
        x = inputs
        if self.augmentation_config.enabled:
            augmentation = keras.Sequential(
                [
                    keras.layers.RandomRotation(
                        self.augmentation_config.rotation_factor,
                        fill_mode="constant",
                        fill_value=0.0,
                    ),
                    keras.layers.RandomTranslation(
                        self.augmentation_config.translation_factor,
                        self.augmentation_config.translation_factor,
                        fill_mode="constant",
                        fill_value=0.0,
                    ),
                    keras.layers.RandomZoom(
                        self.augmentation_config.zoom_factor,
                        fill_mode="constant",
                        fill_value=0.0,
                    ),
                ],
                name="training_augmentation",
            )
            x = augmentation(x)

        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="character")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="airwrite_character_recognizer")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def _require_tensorflow() -> Any:
        try:
            return importlib.import_module("tensorflow")
        except ImportError as error:
            raise TrainingDependencyError(
                "TensorFlow is required to build the character model. "
                "Install training dependencies before training."
            ) from error
