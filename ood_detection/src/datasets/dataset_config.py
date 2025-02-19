from ood_detection.src.datasets.dataset import Dataset

def report_dataset_configuration(in_data_class_instance: object, ood_dataset_ids: list[str], logger):
    """
    Checks the channel dimensions of the in-distribution dataset and each OOD dataset.
    Since channel adaptation is applied later, this function only logs which OOD datasets
    will have their channels adjusted:
      - If an OOD dataset has fewer channels than the in-distribution data, it will be pre-duplicated.
      - If an OOD dataset has more channels, only the first channel(s) will be used.

    Args:
        in_data_class_instance (object): An instance of the in-distribution dataset class.
        ood_dataset_ids (list[str]): List of dataset IDs for the OOD datasets.
        logger: Logger instance for logging messages.

    """
    in_dataset_id = in_data_class_instance.dataset_id
    in_sample_shape = in_data_class_instance.sample_shape
    in_channels = in_sample_shape[0]
    logger.info(f"In-distribution Dataset '{in_dataset_id}' sample shape: {in_sample_shape}")

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
            if ood_channels < in_channels:
                logger.info(
                    f"OOD dataset '{ood_id}' has {ood_channels} channels. It will be pre-duplicated to match "
                    f"the in-distribution channel count of {in_channels}."
                )
            else:  # ood_channels > in_channels
                logger.info(
                    f"OOD dataset '{ood_id}' has {ood_channels} channels. Only the first {in_channels} channel(s) "
                    f"will be used to match the in-distribution data."
                )
        else:
            logger.info(f"OOD dataset '{ood_id}' already has matching channel dimensions: {in_channels} channel(s).")
