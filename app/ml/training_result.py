from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    best_validation_accuracy: float
    model_path: Path
    history_path: Path
