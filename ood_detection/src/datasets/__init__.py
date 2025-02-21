"""A sub-package that contains datasets and algorithms for partitioning datasets for OOD Detection."""

from ood_detection.src.datasets.tissuemnist import TissueMnist
from ood_detection.src.datasets.pathmnist import PathMnist
from ood_detection.src.datasets.chestmnist import ChestMnist
from ood_detection.src.datasets.dermamnist import DermaMnist


DATASET_IDS = [
    TissueMnist.dataset_id,
    PathMnist.dataset_id,
    ChestMnist.dataset_id,
    DermaMnist.dataset_id

]
"""Contains the IDs of all available datasets."""

DEFAULT_DATASET_ID = DATASET_IDS[0]
"""Contains the ID of the default dataset."""

__all__ = [
    'TissuemMist',
    'PathMnist',
    'ChestMnist',
    'DermaMnist'

    'DATASET_IDS',
    'DEFAULT_DATASET_ID'
]
