from collections.abc import Mapping
from typing import Any

from app.inference.case_label_resolver import CaseLabelResolver
from app.inference.exceptions import InvalidModelBundleError
from app.inference.model_bundle import ModelBundle


class ModelBundleValidator:
    ARTIFACT_CONTRACT_KEYS = (
        "input_width",
        "input_height",
        "channels",
        "orientation_transform",
        "intensity_inversion",
        "normalization_divisor",
        "background",
        "foreground",
    )

    def validate(
        self,
        bundle: ModelBundle,
        runtime_preprocessing_contract: Mapping[str, object],
    ) -> None:
        if bundle.task_type != "letter_identity_classification":
            raise InvalidModelBundleError("Model task_type must be letter_identity_classification")
        if bundle.case_sensitive is not False:
            raise InvalidModelBundleError("Identity model metadata must set case_sensitive=false")
        if bundle.case_source != "user_selected_mode":
            raise InvalidModelBundleError("Identity model case_source must be user_selected_mode")
        if bundle.num_classes != 26:
            raise InvalidModelBundleError("Identity model must contain exactly 26 classes")

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
        mapping_lengths = {
            len(bundle.identity_labels),
            len(bundle.lowercase_display_labels),
            len(bundle.uppercase_display_labels),
        }
        if mapping_lengths != {bundle.num_classes}:
            raise InvalidModelBundleError("Model class count does not match label mappings")
        try:
            CaseLabelResolver.validate_mappings(
                bundle.identity_labels,
                bundle.lowercase_display_labels,
                bundle.uppercase_display_labels,
            )
        except ValueError as error:
            raise InvalidModelBundleError(str(error)) from error

        for key in self.ARTIFACT_CONTRACT_KEYS:
            if key not in bundle.preprocessing_contract:
                raise InvalidModelBundleError(f"Missing preprocessing contract field: {key}")

        expected_runtime = {
            "output_width": bundle.preprocessing_contract["input_width"],
            "output_height": bundle.preprocessing_contract["input_height"],
            "channels": bundle.preprocessing_contract["channels"],
            "background": bundle.preprocessing_contract["background"],
            "foreground": bundle.preprocessing_contract["foreground"],
            "normalization_divisor": bundle.preprocessing_contract["normalization_divisor"],
            "orientation_transform": "none",
            "invert_input": bundle.preprocessing_contract["intensity_inversion"],
        }
        for key, expected_value in expected_runtime.items():
            if runtime_preprocessing_contract.get(key) != expected_value:
                raise InvalidModelBundleError(
                    "Model and preprocessing configuration are incompatible: "
                    f"{key}={expected_value!r} expected, "
                    f"{runtime_preprocessing_contract.get(key)!r} runtime"
                )
        if bundle.preprocessing_contract["orientation_transform"] != "transpose":
            raise InvalidModelBundleError("EMNIST source orientation transform must be transpose")

        artifact_shape = (
            bundle.preprocessing_contract["input_height"],
            bundle.preprocessing_contract["input_width"],
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
