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
        use_batch_normalization: bool = False,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
        model_name: str = "airwrite_character_recognizer",
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.augmentation_config.validate()
        self.learning_rate = learning_rate
        if dense_units <= 0:
            raise ValueError("dense_units must be greater than 0")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be between 0 inclusive and 1 exclusive")
        self.use_batch_normalization = use_batch_normalization
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model_name = model_name

    def build(
        self,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 26,
        dropout_rate: float | None = None,
        include_augmentation: bool | None = None,
    ) -> Any:
        if len(input_shape) != 3 or any(dimension <= 0 for dimension in input_shape):
            raise ValueError(
                f"input_shape must contain three positive dimensions, got {input_shape}"
            )
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        selected_dropout = self.dropout_rate if dropout_rate is None else dropout_rate
        if not 0.0 <= selected_dropout < 1.0:
            raise ValueError("dropout_rate must be between 0 inclusive and 1 exclusive")
        augmentation_enabled = (
            self.augmentation_config.enabled
            if include_augmentation is None
            else include_augmentation
        )

        tensorflow = self._require_tensorflow()
        keras = tensorflow.keras
        inputs = keras.Input(shape=input_shape, name="character_image")
        x = inputs
        if augmentation_enabled:
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
        if self.use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        if self.use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        if self.use_batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(self.dense_units, activation="relu")(x)
        x = keras.layers.Dropout(selected_dropout)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="character")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            ],
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
