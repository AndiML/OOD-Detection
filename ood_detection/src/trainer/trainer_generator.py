import logging
from typing import Any, Optional

import torch

from ood_detection.src.trainer.base_trainer import BaseTrainer
from ood_detection.src.experiments.tracker import ExperimentLogger


def create_trainer_from_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    num_epochs: int,
    experiment_path: str,
    training_logger: logging.Logger,
    experiment_logger: ExperimentLogger,
    scheduler: Optional[Any] = None,
    enhanced_ood: bool = False,  # New parameter to toggle enhanced OOD detection.
) -> BaseTrainer:
    """
    Creates a trainer instance from the provided model and trainer parameters.

    If `enhanced_ood` is True and the model id is 'vae', then an instance of the
    EnhancedVAETrainer (which includes the score matching functionality) is returned.
    Otherwise, the trainer is obtained from the registered trainers based on model.model_id.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (str): The device identifier (e.g., 'cpu' or 'cuda').
        num_epochs (int): Number of training epochs.
        experiment_path (str): Directory path where experiment artifacts will be saved.
        training_logger (logging.Logger): Logger used for training-related messages.
        experiment_logger (ExperimentLogger): Logger dedicated to logging experiment details.
        scheduler (Optional[Any]): Learning rate scheduler (if any).
        enhanced_ood (bool): If True, returns an instance of EnhancedVAETrainer for 'vae' models.

    Returns:
        BaseTrainer: An instance of a trainer corresponding to the model's id.

    Raises:
        ValueError: If no trainer is registered for the given model id.
    """
    if enhanced_ood and model.model_id == 'vae':
        from ood_detection.src.trainer.variational_autoencoder_trainer_score import EnhancedVAETrainer
        trainer_class = EnhancedVAETrainer
    else:
        trainer_class = BaseTrainer._registry.get(model.model_id)
        if trainer_class is None:
            raise ValueError(f"No trainer registered for model id '{model.model_id}'.")

    return trainer_class(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        experiment_path=experiment_path,
        training_logger=training_logger,
        experiment_logger=experiment_logger,
        scheduler=scheduler
    )
