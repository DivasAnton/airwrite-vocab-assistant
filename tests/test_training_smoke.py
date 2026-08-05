import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from app.ml.character_model_builder import AugmentationConfig, CharacterModelBuilder
from app.ml.character_model_trainer import CharacterModelTrainer

RUN_SMOKE_TEST = os.getenv("RUN_TRAINING_SMOKE_TESTS", "false").lower() == "true"


@pytest.mark.skipif(
    not RUN_SMOKE_TEST or importlib.util.find_spec("tensorflow") is None,
    reason="Set RUN_TRAINING_SMOKE_TESTS=true with TensorFlow installed",
)
def test_one_epoch_training_creates_checkpoint_and_history(tmp_path: Path) -> None:
    import tensorflow as tf

    images = np.zeros((52, 28, 28, 1), dtype=np.float32)
    labels = np.repeat(np.arange(26, dtype=np.int64), 2)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(26)
    model_path = tmp_path / "model.keras"
    history_path = tmp_path / "history.csv"
    trainer = CharacterModelTrainer(
        builder=CharacterModelBuilder(AugmentationConfig(enabled=False)),
        model_path=model_path,
        history_path=history_path,
        max_epochs=1,
    )

    result = trainer.train(dataset, dataset, verbose=0)

    assert result.epochs_completed == 1
    assert model_path.exists()
    assert history_path.exists()
