"""A sub-package that contains datasets and algorithms for partitioning datasets for OOD Detection."""

from ood_detection.src.datasets.mnist import Mnist
from ood_detection.src.datasets.tissuemnist import TissueMnist
from ood_detection.src.datasets.pathmnist import PathMnist
from ood_detection.src.datasets.chestmnist import ChestMnist
from ood_detection.src.datasets.dermamnist import DermaMnist
from ood_detection.src.datasets.fmnist import Fmnist
from ood_detection.src.datasets.cifar10 import Cifar10
from ood_detection.src.datasets.cinic10 import Cinic10
from ood_detection.src.datasets.cifar100 import Cifar100
from ood_detection.src.datasets.cifar100_super import Cifar100Super
from ood_detection.src.datasets.tiny_imagenet import Imagenet

DATASET_IDS = [
    Cifar100.dataset_id,
    Cifar10.dataset_id,
    Cifar100Super.dataset_id,
    Imagenet.dataset_id,
    Cinic10.dataset_id,
    Fmnist.dataset_id,
    Mnist.dataset_id,
    TissueMnist.dataset_id,
    PathMnist.dataset_id,
    ChestMnist.dataset_id,
    DermaMnist.dataset_id

]
"""Contains the IDs of all available datasets."""

DEFAULT_DATASET_ID = Mnist.dataset_id
"""Contains the ID of the default dataset."""

__all__ = [
    'Mnist',
    'Tissuemnist',
    'Fmnist',
    'Cifar10',
    'Cinic10',
    'Cifar100'
    'Cifar100-super',
    'Imagenet'
    'Dataset',

    'DATASET_IDS',
    'DEFAULT_DATASET_ID'
]
