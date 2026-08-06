import csv
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.ml.exceptions import ModelEvaluationError
from app.ml.letter_identity_labels import LETTER_IDENTITY_LABELS, identity_to_index
from app.ml.model_evaluator import EvaluationResult, ModelEvaluator
from app.preprocessing.handwriting_preprocessor import HandwritingPreprocessor
from app.preprocessing.model_input_preprocessor import ModelInputPreprocessor
from app.preprocessing.preprocessing_source import PreprocessingSource


@dataclass(frozen=True)
class AirWriteIdentitySample:
    image_path: Path
    identity: str
    writing_style: str
    session_id: str


class AirWriteIdentityEvaluator:
    VALID_STYLES = frozenset({"uppercase", "lowercase"})

    def __init__(
        self,
        preprocessor: HandwritingPreprocessor | None = None,
        model_input_preprocessor: ModelInputPreprocessor | None = None,
        images_preprocessed: bool = True,
    ) -> None:
        self.preprocessor = preprocessor or HandwritingPreprocessor()
        self.model_input_preprocessor = model_input_preprocessor or ModelInputPreprocessor()
        self.images_preprocessed = images_preprocessed
        self.evaluator = ModelEvaluator(LETTER_IDENTITY_LABELS)

    def load_manifest(self, path: Path) -> list[AirWriteIdentitySample]:
        if not path.is_file():
            raise ModelEvaluationError(f"AirWrite evaluation manifest does not exist: {path}")
        samples: list[AirWriteIdentitySample] = []
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            required = {"image_path", "identity", "writing_style", "session_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ModelEvaluationError(
                    "AirWrite manifest requires image_path, identity, writing_style, session_id"
                )
            for row in reader:
                identity = row["identity"].strip().lower()
                style = row["writing_style"].strip().lower()
                if identity not in LETTER_IDENTITY_LABELS:
                    raise ModelEvaluationError(f"Invalid AirWrite identity: {identity!r}")
                if style not in self.VALID_STYLES:
                    raise ModelEvaluationError(f"Invalid writing_style: {style!r}")
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                if not image_path.is_file():
                    raise ModelEvaluationError(
                        f"AirWrite evaluation image is missing: {image_path}"
                    )
                samples.append(
                    AirWriteIdentitySample(
                        image_path=image_path,
                        identity=identity,
                        writing_style=style,
                        session_id=row["session_id"].strip(),
                    )
                )
        if not samples:
            raise ModelEvaluationError("AirWrite evaluation manifest is empty")
        return samples

    def evaluate(self, model: object, samples: list[AirWriteIdentitySample]) -> EvaluationResult:
        predict_on_batch = getattr(model, "predict_on_batch", None)
        if not callable(predict_on_batch):
            raise ModelEvaluationError("Model must provide predict_on_batch")
        images = np.stack([self._prepare_image(sample.image_path) for sample in samples]).astype(
            np.float32
        )[..., np.newaxis]
        probabilities = np.asarray(predict_on_batch(images), dtype=np.float32)
        y_true = np.asarray(
            [identity_to_index(sample.identity) for sample in samples], dtype=np.int64
        )
        base = self.evaluator.evaluate_predictions(y_true, probabilities)
        y_pred = np.argmax(probabilities, axis=1)
        style_metrics: dict[str, dict[str, float]] = {}
        for style in sorted(self.VALID_STYLES):
            indices = [
                index for index, sample in enumerate(samples) if sample.writing_style == style
            ]
            if indices:
                style_metrics[style] = {
                    "accuracy": float(np.mean(y_true[indices] == y_pred[indices])),
                    "sample_count": float(len(indices)),
                }
        errors: list[dict[str, object]] = []
        for index, sample in enumerate(samples):
            if y_pred[index] == y_true[index]:
                continue
            top_indices = np.argsort(probabilities[index])[-3:][::-1]
            errors.append(
                {
                    "image_path": sample.image_path.as_posix(),
                    "true_label": sample.identity,
                    "predicted_label": LETTER_IDENTITY_LABELS[int(y_pred[index])],
                    "confidence": float(probabilities[index, y_pred[index]]),
                    "top_3": [
                        {
                            "label": LETTER_IDENTITY_LABELS[int(class_index)],
                            "probability": float(probabilities[index, class_index]),
                        }
                        for class_index in top_indices
                    ],
                    "source": sample.writing_style,
                    "session_id": sample.session_id,
                }
            )
        return replace(
            base,
            source_metrics=style_metrics,
            error_analysis=errors,
            probabilities=probabilities.astype(float).tolist(),
            true_indices=y_true.astype(int).tolist(),
            image_paths=[sample.image_path.as_posix() for sample in samples],
            sources=[sample.writing_style for sample in samples],
        )

    def _prepare_image(self, image_path: Path) -> np.ndarray:
        if not self.images_preprocessed:
            return self.preprocessor.process_file(image_path).normalized_image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ModelEvaluationError(f"Could not decode AirWrite evaluation image: {image_path}")
        return self.model_input_preprocessor.process(
            image,
            source=PreprocessingSource.AIRWRITE_CANVAS,
            orientation_transform="none",
        ).normalized_image
