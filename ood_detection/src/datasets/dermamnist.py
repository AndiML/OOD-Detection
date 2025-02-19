"""Represents a module containing DermaMNIST dataset."""

import numpy
import torch
import torchvision  # type: ignore

from medmnist import DermaMNIST
from ood_detection.src.datasets.dataset import Dataset, DatasetData


class DermaMnist(Dataset):
    """Represents the  DermaMNIST dataset."""

    dataset_id = 'dermamnist'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str, load_data: bool = True) -> None:
        """
        Initializes a new DermaMNIST instance.

        Args:
            path (str): The path where the DermaMNIST dataset is stored.
            load_data (bool): If True, download and load the data. If False, only set metadata.
        """
        self.path = path
        self.name = 'DermaMNIST'
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.76311266,), (0.13660511,))
        ])

        # If load_data is True, load the full dataset. Otherwise, skip it.
        if load_data:
            self._load_datasets()
        else:
            # Set empty placeholders if data isn't loaded.
            self._training_data = None
            self._validation_data = None
            self._test_data = None
            self._training_labels = []
            self._validation_labels = []
            self._test_labels = []

    def _load_datasets(self) -> None:
        """Loads all data."""
        # Load training data.
        self._training_data = DermaMNIST(
            root=self.path, split='train', download=True, transform=self.transform
        )
        self._training_data.labels = [label[0] for label in self._training_data.labels]
        self._training_labels = self._training_data.labels

        # Load validation data.
        self._validation_data = DermaMNIST(
            root=self.path, split='val', download=True, transform=self.transform
        )
        self._validation_data.labels = [label[0] for label in self._validation_data.labels]
        self._validation_labels = self._validation_data.labels

        # Load test data.
        self._test_data = DermaMNIST(
            root=self.path, split='test', download=True, transform=self.transform
        )
        self._test_data.labels = [label[0] for label in self._test_data.labels]
        self._test_labels = self._test_data.labels


    def get_training_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for training.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._training_labels

    def get_validation_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for validation.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._validation_labels

    def get_test_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for testing.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self._test_labels

    @property
    def training_data(self) -> DatasetData:
        """Gets the training data of the dataset.

        Returns:
            DatasetData: Returns the training data of the dataset.
        """

        return self._training_data

    @property
    def validation_data(self) -> DatasetData:
        """Gets the validation data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._validation_data

    @property
    def test_data(self) -> DatasetData:
        """Gets the test data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._test_data

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """
        return tuple(3, 28, 28)

    @property
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Returns:
            int: Returns the number of distinct classes.
        """
        return len(numpy.unique(self._test_labels))

