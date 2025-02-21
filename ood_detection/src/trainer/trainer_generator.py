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
) -> BaseTrainer:
    """
    Creates a trainer instance from the provided model and trainer parameters.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (str): The device identifier (e.g., 'cpu' or 'cuda').
        num_epochs (int): Number of training epochs.
        experiment_path (str): Directory path where experiment artifacts will be saved.
        training_logger: Logger used for training-related messages.
        experiment_logger: Logger dedicated to logging experiment details.
        scheduler: Learning rate scheduler (if any).

    Returns:
        BaseTrainer: An instance of a trainer that is registered for the model's id.

    Raises:
        ValueError: If no trainer is registered for the given model id.
    """
    trainer_instance = BaseTrainer._registry.get(model.model_id)
    if trainer_instance is None:
        raise ValueError(f"No trainer registered for model id '{model.model_id}'.")
    return trainer_instance(
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
