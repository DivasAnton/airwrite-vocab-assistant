from collections.abc import Mapping
from typing import Any

from app.inference.exceptions import InvalidModelBundleError
from app.inference.model_bundle import ModelBundle


class ModelBundleValidator:
    CONTRACT_KEYS = (
        "output_width",
        "output_height",
        "channels",
        "background",
        "foreground",
        "normalized_min",
        "normalized_max",
        "content_width",
        "content_height",
        "binary_threshold",
        "crop_padding",
        "min_foreground_pixels",
        "invert_input",
        "center_of_mass",
    )

    def validate(
        self,
        bundle: ModelBundle,
        runtime_preprocessing_contract: Mapping[str, object],
    ) -> None:
        model_input_shape = self._single_shape(bundle.model.input_shape, "input")
        if model_input_shape != bundle.expected_input_shape:
            raise InvalidModelBundleError(
                "Model input shape does not match metadata: "
                f"{model_input_shape} != {bundle.expected_input_shape}"
            )

        model_output_shape = self._single_shape(bundle.model.output_shape, "output")
        if len(model_output_shape) != 1 or model_output_shape[0] != bundle.num_classes:
            raise InvalidModelBundleError(
                f"Model output shape {model_output_shape} does not match "
                f"{bundle.num_classes} classes"
            )
        if bundle.num_classes != len(bundle.labels):
            raise InvalidModelBundleError("Model class count does not match label count")

        for key in self.CONTRACT_KEYS:
            if (
                key not in bundle.preprocessing_contract
                or key not in runtime_preprocessing_contract
            ):
                raise InvalidModelBundleError(f"Missing preprocessing contract field: {key}")
            if bundle.preprocessing_contract[key] != runtime_preprocessing_contract[key]:
                raise InvalidModelBundleError(
                    "Model and preprocessing configuration are incompatible: "
                    f"{key}={bundle.preprocessing_contract[key]!r} artifact, "
                    f"{runtime_preprocessing_contract[key]!r} runtime"
                )

        artifact_shape = (
            bundle.preprocessing_contract["output_height"],
            bundle.preprocessing_contract["output_width"],
            bundle.preprocessing_contract["channels"],
        )
        if artifact_shape != bundle.expected_input_shape:
            raise InvalidModelBundleError(
                "Preprocessing output dimensions do not match model metadata input shape"
            )

    @staticmethod
    def _single_shape(shape: Any, shape_name: str) -> tuple[int, ...]:
        if isinstance(shape, list):
            raise InvalidModelBundleError(f"Multiple model {shape_name}s are not supported")
        try:
            values = tuple(shape)
        except TypeError as error:
            raise InvalidModelBundleError(f"Model {shape_name} shape is unavailable") from error
        if len(values) < 2 or values[0] is not None:
            raise InvalidModelBundleError(
                f"Model {shape_name} shape must include a dynamic batch dimension"
            )
        dimensions = values[1:]
        if not all(isinstance(value, int) and value > 0 for value in dimensions):
            raise InvalidModelBundleError(
                f"Model {shape_name} dimensions must be positive integers"
            )
        return tuple(dimensions)
