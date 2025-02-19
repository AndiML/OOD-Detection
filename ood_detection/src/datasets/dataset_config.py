from ood_detection.src.datasets.dataset import Dataset

def check_dataset_configuration(in_data_class_instance: object, ood_dataset_ids: list[str], logger) -> list[str]:
    """
    Checks that the in-distribution dataset and each OOD dataset have matching channel dimensions.
    Incompatible OOD datasets are skipped. If none of the provided OOD datasets are compatible,
    a ValueError is raised.

    Args:
        in_data_class_instance (object): An instance of the in-distribution dataset class.
        ood_dataset_ids (list[str]): List of dataset IDs for the OOD datasets.
        dataset_path (str): Path to the dataset directory.
        logger: Logger instance for logging messages.

    Returns:
        list[str]: A list of OOD dataset IDs that have matching channel dimensions.

    Raises:
        ValueError: If none of the provided OOD datasets have matching channel dimensions.
    """
    in_dataset_id = in_data_class_instance.dataset_id
    in_sample_shape = in_data_class_instance.sample_shape
    in_channels = in_sample_shape[0]
    logger.info(f"In-distribution Dataset '{in_dataset_id}' sample shape: {in_sample_shape}")

    valid_ood_ids = []
    for ood_id in ood_dataset_ids:
        try:
            # Create OOD dataset instance in metadata-only mode.
            ood_dataset = Dataset.create(ood_id, "./", load_data=False)
        except Exception as e:
            logger.error(f"Could not create OOD dataset '{ood_id}': {e}")
            continue

        ood_sample_shape = ood_dataset.sample_shape
        ood_channels = ood_sample_shape[0]
        logger.info(f"OOD Dataset '{ood_id}' sample shape: {ood_sample_shape}")

        if ood_channels != in_channels:
            logger.error(
                f"Channel mismatch: In-distribution dataset '{in_dataset_id}' has {in_channels} channels, "
                f"but OOD dataset '{ood_id}' has {ood_channels} channels. Skipping this OOD dataset."
            )
        else:
            valid_ood_ids.append(ood_id)

    if not valid_ood_ids:
        msg = "No OOD datasets with matching channel dimensions were found."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Valid OOD datasets for Comparison: {valid_ood_ids if len(valid_ood_ids) > 1 else valid_ood_ids[0]}")
    return valid_ood_ids
