import logging
import collections
import os
import torch

from tqdm import tqdm

from abc import ABC, abstractmethod
from ood_detection.src.experiments.tracker import ExperimentLogger

class BaseTrainer(ABC):
    """
    A base trainer class that implements a generic training and validation loop,
    logging of statistics via both a Python logger and an ExperimentLogger,
    model checkpointing, and integration of learning rate schedulers.
    Automatically registers subclasses based on a trainer identifier.
    Inherited classes need to implement model-specific training and validation logic.
    """

    _registry: dict[str, type["BaseTrainer"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically register subclasses that define a trainer_id attribute.
        if hasattr(cls, "trainer_id"):
            trainer_id = getattr(cls, "trainer_id")
            cls._registry[trainer_id] = cls

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_epochs: int,
        experiment_path: str,
        training_logger: logging.Logger,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        experiment_logger: ExperimentLogger = None):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimizer.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Device to run on.
            num_epochs (int): Number of training epochs.
            best_model_dir: str
            training_logger (logging.Logger): The logger for the training process.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            experiment_logger (ExperimentLogger, optional): An instance of ExperimentLogger to log metrics.
        """

        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.num_train_batches = len(train_loader)
        self.val_loader = val_loader
        self.num_val_batches = len(val_loader)
        self.device = device
        self.num_epochs = num_epochs
        self.model_directory = os.path.join(experiment_path, "model-checkpoint")
        self.experiment_path = experiment_path
        self.training_logger = training_logger
        self.scheduler = scheduler
        self.experiment_logger = experiment_logger
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    @abstractmethod
    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) ->  dict[str, float]:
        """
        Performs a single training step (forward, loss computation, backward, optimizer step)
        for a given batch. Should return the loss value for logging purposes.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data.

        Returns:
            dict[str, float]: The computed loss for the batch.
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) ->  dict[str, float]:
        """
        Performs a single validation step (forward pass and loss computation)
        for a given batch. Should return the loss value for logging purposes.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of data.

        Returns:
            dict[str, float]: The computed loss for the batch.
        """
        raise NotImplementedError


    def train(self):
        """Executes the entire training loop."""
        self.training_logger.info("Starting training for %d epochs.", self.num_epochs,  extra={'start_section': True})

        for epoch in range(1, self.num_epochs + 1):
            if self.experiment_logger is not None:
                self.experiment_logger.train()
                self.experiment_logger.begin_epoch(epoch)

            # Sets model into training mode
            self.model.train()

            # Use a dictionary to accumulate sums of each metric
            running_metrics = collections.defaultdict(float)

            with tqdm(total=self.num_train_batches, desc=f"Epoch {epoch}/{self.num_epochs}", unit="batch") as pbar:
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    metrics = self.train_step(batch)
                    self.optimizer.step()

                    # Accumulate each metric
                    for key, value in metrics.items():
                        running_metrics[key] += value

                    # Updates the  progress bar
                    pbar.update(1)
                    pbar.set_postfix({k: f"{v / batch_idx:.4f}" for k, v in running_metrics.items()})

            # Computes average metrics for the entire epoch
            epoch_metrics = {
                k: v / self.num_train_batches for k, v in running_metrics.items()
            }
            self.training_logger.info("Epoch %d Average Training Metrics: %s", epoch, epoch_metrics,  extra={'end_section': True})

            # Log training metrics via the experiment logger if provided
            if self.experiment_logger is not None:
                self.experiment_logger.log_model_metrics(epoch_metrics)

            # Validate model
            self.validate(epoch)


    def validate(self, current_epoch: int):

        # Sets model into validation mode
        self.model.eval()

        with torch.no_grad():

            # Initialize a defaultdict for accumulating metrics, if not already defined.
            running_metrics = collections.defaultdict(float)

            with tqdm(total=self.num_val_batches, desc="Validation", unit="batch") as pbar:
                for batch in self.val_loader:
                    metrics = self.validation_step(batch)
                    for key, value in metrics.items():
                        running_metrics[key] += value

                    # Update progress bar after processing each batch.
                    pbar.update(1)

        # Average the metrics
        avg_metrics = {
            k: v / self.num_val_batches for k, v in running_metrics.items()
        }
        # Step scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step(avg_metrics['loss'])

        self.training_logger.info("Average Validation Metrics: %s", avg_metrics)

        if self.experiment_logger is not None:
            # Switch to validation phase
            self.experiment_logger.val()
            self.experiment_logger.log_model_metrics(avg_metrics)
            self.experiment_logger.end_epoch()

        # Check if the current validation loss is the best so far.
        current_val_loss = avg_metrics['loss']
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                # Delete the previous best model if it exists.
                if hasattr(self, 'best_model_path') and self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                    self.training_logger.info("Deleted previous best model")

                self.best_val_loss = current_val_loss
                self.best_epoch = current_epoch
                # Annotate the filename with the epoch and loss value
                best_model_filename = f"best_model_epoch_{current_epoch}_loss_{round(current_val_loss,2):.2f}.pt"
                self.best_model_path = os.path.join(self.model_directory, best_model_filename)
                self.save_model(self.best_model_path)
                self.training_logger.info(
                    "Best model updated at epoch %d with validation loss: %.2f",
                    current_epoch, current_val_loss, extra={'end_section': True}
                )

        # Switch back to training mode
        self.model.train()


    def save_model(self, path: str):
        """
        Saves the model state to the given path.

        Args:
            path (str): The path where the model state should be saved.
        """
        torch.save(self.model.state_dict(), path)
        self.training_logger.info("Model saved to %s", path)


    def save_checkpoint(self, checkpoint_path: str, epoch: int) -> None:

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'experiment_logger_state': (
                self.experiment_logger.get_state() if self.experiment_logger else None
            ),
        }
        torch.save(checkpoint, checkpoint_path)
        self.training_logger.info("Checkpoint saved to %s at epoch %d", checkpoint_path, epoch)


    def load_checkpoint(self, checkpoint_path: str) -> int:

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.experiment_logger and checkpoint.get('experiment_logger_state'):
            self.experiment_logger.set_state(checkpoint['experiment_logger_state'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch.
        self.training_logger.info("Checkpoint loaded from %s. Resuming at epoch %d", checkpoint_path, start_epoch)

        return start_epoch
