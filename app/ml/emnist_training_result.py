from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EMNISTTrainingResult:
    experiment_id: str
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    best_validation_top_3_accuracy: float
    checkpoint_path: Path
    history_path: Path
