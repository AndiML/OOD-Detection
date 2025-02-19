import torch
from torch.utils.data import Subset, ConcatDataset

from ood_detection.src.datasets.dataset import Dataset, DatasetData
from ood_detection.src.datasets.transformer import OODTransformer, TransformWrapperOOD

class DataPartitioner:
    """
    Partitions the test set of an in-distribution dataset for OOD experiments.
    In the combined dataset, in-distribution samples are labeled as 1 ("known")
    and OOD samples are labeled as 0 ("unknown").

    Supported methods:
      - partition_inliers: Use the first n distinct classes as in-distribution and all others as OOD.
      - partition_external_ood: Use an external dataset as the OOD test set.
    """
    def __init__(self, in_dataset: DatasetData):
        """
        Args:
            in_dataset: An instance of a Dataset subclass (e.g. TissueMnist) representing in-distribution data.
        """
        self.in_dataset = in_dataset
        # Define target transforms: known → label 1, unknown → label 0.
        self.known_transform = lambda target: 1
        self.unknown_transform = lambda target: 0
        self.combined_data = None

    def _filter_dataset_by_classes(self, dataset: torch.utils.data.Dataset, classes: set, include: bool) -> Subset:
        """
        Filters a dataset by its labels (assumes dataset.labels is a list where each element is a label).

        Args:
            dataset: The dataset to filter.
            classes: A set of labels to include or exclude.
            include: If True, only include samples whose label is in classes.
                     If False, include samples whose label is NOT in classes.
        """
        filtered_indices = [
            idx for idx, label in enumerate(dataset.labels)
            if (label in classes if include else label not in classes)
        ]
        return Subset(dataset, filtered_indices)

    def _combine(self, known_subset: torch.utils.data.Dataset, unknown_subset: torch.utils.data.Dataset):
        """
        Wraps the known and unknown subsets with the label transformation and then concatenates them.
        """
        desired_channels = self.in_dataset.sample_shape[0]
        # Create a combined transformer for known and unknown samples.
        known_transformer = OODTransformer(desired_channels=desired_channels, mode="known")
        unknown_transformer = OODTransformer(desired_channels=desired_channels, mode="unknown")
        # Wrap the subsets with our generic wrapper.
        known_dataset = TransformWrapperOOD(known_subset, known_transformer)
        unknown_dataset = TransformWrapperOOD(unknown_subset, unknown_transformer)
        self.combined_data = ConcatDataset([known_dataset, unknown_dataset])


    def partition_inliers(self, n_inliers: int):
        """
        Partitions the in-distribution test data by selecting the first n distinct classes (sorted)
        as known (inliers), with all other classes considered unknown (OOD).

        Args:
            n_inliers (int): The number of distinct classes to use as inliers.

        Raises:
            ValueError: if n_inliers is greater than the total number of classes in the dataset.
        """
        # Determine the distinct classes present in the dataset (assuming labels are comparable)
        distinct_classes = sorted(set(self.in_dataset.test_data.labels))
        if n_inliers > len(distinct_classes):
            raise ValueError(f"Requested {n_inliers} inlier classes, but only {len(distinct_classes)} are available.")

        inlier_classes = set(distinct_classes[:n_inliers])

        known_subset = self._filter_dataset_by_classes(self.in_dataset.test_data, inlier_classes, include=True)
        unknown_subset = self._filter_dataset_by_classes(self.in_dataset.test_data, inlier_classes, include=False)
        self._combine(known_subset, unknown_subset)

    def partition_external_ood(self, ood_dataset_id: str, dataset_path):
        """
        Uses the in-distribution test set as known data and an external dataset (retrieved from the registry)
        as unknown OOD data.
        """
        ood_dataset = Dataset.create(ood_dataset_id, dataset_path)
        known_dataset = self.in_dataset.test_data
        unknown_dataset = ood_dataset.test_data
        self._combine(known_dataset, unknown_dataset)


    def partition(self, partition_method: str, num_inliers: int = 1,
                  ood_dataset_id: str = None, dataset_path: str = None) -> None:
        """
        Dispatches to the correct partitioning method.

        Args:
            partition_method (str): Either 'internal' or 'external'.
            n_inliers (int, optional): Number of inlier classes (required for internal partitioning).
            ood_dataset_id (str, optional): Identifier for the external dataset (required for external partitioning).
            dataset_path (str, optional): Path to the external dataset (required for external partitioning).

        Raises:
            ValueError: if required parameters are missing or partition_method is unrecognized.
        """
        method = partition_method.lower()
        if method == 'internal':
            self.partition_inliers(num_inliers)
        elif method == 'external':
            self.partition_external_ood(ood_dataset_id, dataset_path)
        else:
            raise ValueError(f"Unrecognized partitioning method: {partition_method}")

    def get_dataloader(self,
                       batch_size: int,
                       shuffle: bool = False,
                       num_workers: int = 0,
                       pin_memory: bool = True) -> torch.utils.data.DataLoader:
        """
        Wraps the combined dataset in a DataLoader.
        """
        if self.combined_data is None:
            raise ValueError("No combined data available. Please partition your data first "
                             "(e.g., call partition() with the appropriate method).")
        return torch.utils.data.DataLoader(
            dataset=self.combined_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
