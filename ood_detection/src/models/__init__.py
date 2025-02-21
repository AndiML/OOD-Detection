"""A sub-package that contains models and algorithms retrieving a model that is trained on In-Data. """

from ood_detection.src.models.autoencoder import AutoDynamicAutoencoder
from ood_detection.src.models.variational_autoencoder import AutoDynamicVariationalAutoencoder


"""Contains the IDs of all available model architectures."""
MODEL_IDS = [
    AutoDynamicAutoencoder.model_id,
    AutoDynamicVariationalAutoencoder.model_id
]

"""Contains the ID of the default model architecture."""
DEFAULT_MODEL_ID = MODEL_IDS[0]

__all__ = [
    'MODEL_IDS',
    'DEFAULT_MODEL_ID'
]
