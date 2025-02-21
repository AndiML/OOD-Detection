"""A sub-package that retrieves the trainer for a model. """

from ood_detection.src.trainer.autoencoder_trainer import AETrainer
from ood_detection.src.trainer.variational_autoencoder_trainer import VAETrainer

"""Contains the IDs of all available model architectures."""
MODEL_IDS = [
    AETrainer.trainer_id,
    VAETrainer.trainer_id
]

"""Contains the ID of the default model architecture."""
DEFAULT_MODEL_ID = MODEL_IDS[0]

__all__ = [
    'MODEL_IDS',
    'DEFAULT_MODEL_ID'
]
