import importlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.character_model_builder import CharacterModelBuilder
from app.ml.emnist_training_result import EMNISTTrainingResult
from app.ml.exceptions import TrainingDependencyError
from app.ml.training_result import TrainingResult


class CharacterModelTrainer:
    def __init__(
        self,
        builder: CharacterModelBuilder,
        model_path: Path,
        history_path: Path,
        input_shape: tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 26,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        random_seed: int = 42,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be greater than or equal to 0")
        self.builder = builder
        self.model_path = model_path
        self.history_path = history_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed

    def train(
        self, train_dataset: Any, validation_dataset: Any, verbose: int = 1
    ) -> TrainingResult:
        tensorflow = self._require_tensorflow()
        self._set_seeds(tensorflow)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.builder.build(self.input_shape, self.num_classes)
        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ),
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path),
                monitor="val_loss",
                save_best_only=True,
            ),
            tensorflow.keras.callbacks.CSVLogger(str(self.history_path)),
            tensorflow.keras.callbacks.TerminateOnNaN(),
        ]
        history_object = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            shuffle=False,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        history = history_object.history
        validation_losses = [float(value) for value in history.get("val_loss", [])]
        validation_accuracies = [float(value) for value in history.get("val_accuracy", [])]
        if not validation_losses:
            raise RuntimeError("Training history does not contain val_loss")
        best_index = int(np.argmin(validation_losses))
        best_accuracy = (
            validation_accuracies[best_index]
            if best_index < len(validation_accuracies)
            else float("nan")
        )
        return TrainingResult(
            best_epoch=best_index + 1,
            epochs_completed=len(validation_losses),
            best_validation_loss=validation_losses[best_index],
            best_validation_accuracy=best_accuracy,
            model_path=self.model_path,
            history_path=self.history_path,
        )

    def _set_seeds(self, tensorflow: Any) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tensorflow.keras.utils.set_random_seed(self.random_seed)

    @staticmethod
    def _require_tensorflow() -> Any:
        try:
            return importlib.import_module("tensorflow")
        except ImportError as error:
            raise TrainingDependencyError(
                "TensorFlow is required to train the character model. "
                "Install training dependencies before training."
            ) from error


class EMNISTModelTrainer:
    def __init__(
        self,
        builder: CharacterModelBuilder,
        experiment_id: str,
        checkpoint_path: Path,
        history_path: Path,
        max_epochs: int = 40,
        early_stopping_patience: int = 6,
        reduce_lr_patience: int = 3,
        random_seed: int = 42,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0")
        if early_stopping_patience < 0 or reduce_lr_patience < 0:
            raise ValueError("Callback patience values must be non-negative")
        self.builder = builder
        self.experiment_id = experiment_id
        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.random_seed = random_seed

    def train(
        self, train_dataset: Any, validation_dataset: Any, verbose: int = 1
    ) -> tuple[Any, EMNISTTrainingResult]:
        tensorflow = CharacterModelTrainer._require_tensorflow()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tensorflow.keras.utils.set_random_seed(self.random_seed)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.builder.build((28, 28, 1), 26)
        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ),
            tensorflow.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=self.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-6,
            ),
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            ),
            tensorflow.keras.callbacks.CSVLogger(str(self.history_path)),
            tensorflow.keras.callbacks.TerminateOnNaN(),
        ]
        history_object = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        history = history_object.history
        losses = [float(value) for value in history.get("val_loss", [])]
        if not losses:
            raise RuntimeError("Training history does not contain val_loss")
        best_index = int(np.argmin(losses))
        accuracies = [float(value) for value in history.get("val_accuracy", [])]
        top3_values = [float(value) for value in history.get("val_top_3_accuracy", [])]
        best_model = tensorflow.keras.models.load_model(self.checkpoint_path)
        result = EMNISTTrainingResult(
            experiment_id=self.experiment_id,
            epochs_completed=len(losses),
            best_epoch=best_index + 1,
            best_validation_loss=losses[best_index],
            best_validation_accuracy=accuracies[best_index],
            best_validation_top_3_accuracy=top3_values[best_index],
            checkpoint_path=self.checkpoint_path,
            history_path=self.history_path,
        )
        return best_model, result
