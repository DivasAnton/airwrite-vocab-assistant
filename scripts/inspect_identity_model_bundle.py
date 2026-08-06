from app.inference.model_bundle_loader import ModelBundleLoader
from app.inference.model_bundle_validator import ModelBundleValidator
from app.utils.config import identity_model_settings, settings


def main() -> None:
    identity_model_settings.validate()
    bundle = ModelBundleLoader(
        model_path=identity_model_settings.model_path,
        identity_labels_path=identity_model_settings.identity_labels_path,
        lowercase_display_labels_path=identity_model_settings.lowercase_display_labels_path,
        uppercase_display_labels_path=identity_model_settings.uppercase_display_labels_path,
        metadata_path=identity_model_settings.metadata_path,
        preprocessing_config_path=identity_model_settings.preprocessing_config_path,
    ).load()
    ModelBundleValidator().validate(bundle, settings.preprocessing_runtime_contract())

    print(f"Model version: {bundle.model_version}")
    print(f"Task type: {bundle.task_type}")
    print(f"Case-sensitive: {str(bundle.case_sensitive).lower()}")
    print(f"Case source: {bundle.case_source}")
    print(f"Input shape: {bundle.model.input_shape}")
    print(f"Output shape: {bundle.model.output_shape}")
    print(f"Identity labels: {','.join(bundle.identity_labels)}")
    print(f"Lowercase display labels: {','.join(bundle.lowercase_display_labels)}")
    print(f"Uppercase display labels: {','.join(bundle.uppercase_display_labels)}")
    print(f"Preprocessing contract: {bundle.preprocessing_contract}")


if __name__ == "__main__":
    main()
