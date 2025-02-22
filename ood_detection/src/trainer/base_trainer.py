import logging
import collections
import os
import torch

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from abc import ABC, abstractmethod
from ood_detection.src.experiments.tracker import ExperimentLogger

class BaseTrainer(ABC):
    """
    A base trainer class that implements a generic training and validation loop,
    logging of statistics via both a Python logger and an ExperimentLogger,
    model checkpointing, and integration of learning rate schedulers.
    Automatically registers subclasses based on a trainer identifier.
    inherited classes need to implement model-specific training and validation logic.
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

    @abstractmethod
    def post_validation(self, last_batch: tuple[torch.Tensor, torch.Tensor], epoch: int):
        """
        Hook to allow subclasses to process the last batch after the validation phase.

        This method is called at the end of the validation loop, providing the last
        batch processed and the current epoch number.

        Args:
            last_batch (tuple[torch.Tensor, torch.Tensor]): The last batch of data processed
                during validation.
            epoch (int): The current epoch number.
        """
        raise NotImplementedError


    @abstractmethod
    def compute_ood_score_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Computes OOD score for each batch.
        Returns:
            A 1D tensor of ood scores (one per sample).
        """

        raise NotImplementedError


    def compute_ood_scores(self, test_loader):
        """
        Iterates over a test DataLoader and computes anomaly scores and labels.

        Args:
            test_loader (torch.utils.data.Dataloader): The DataLoader to iterate over.

        Returns:
            tuple: (all_scores, all_labels) as concatenated tensors.
        """
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x, labels in test_loader:
                x = x.to(self.device)
                anomaly_scores = self.compute_ood_score_batch(x)
                all_scores.append(anomaly_scores.cpu())
                all_labels.append(labels.cpu())

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        return all_scores, all_labels

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

            # Use Rich progress bar for the training loop.
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True
            ) as progress:
                task = progress.add_task(f"Epoch {epoch}/{self.num_epochs}", total=self.num_train_batches)
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    metrics = self.train_step(batch)
                    self.optimizer.step()

                    # Accumulate metrics
                    for key, value in metrics.items():
                        running_metrics[key] += value

                    # Build a string for average metrics so far
                    avg_metrics_str = " ".join(f"{k}:{(v / batch_idx):.4f}" for k, v in running_metrics.items())
                    # Update progress bar: advance one batch and update description with metrics.
                    progress.update(task, advance=1, description=f"Epoch {epoch}/{self.num_epochs} | {avg_metrics_str}")
                    break

            # Computes average metrics for the entire epoch
            epoch_metrics = {
                k: v / self.num_train_batches for k, v in running_metrics.items()
            }
            self.training_logger.info("Epoch %d Average Training Metrics: %s", epoch, epoch_metrics)

            # Log training metrics via the experiment logger if provided
            self.experiment_logger.begin_epoch(epoch)
            self.experiment_logger.log_model_metrics(epoch_metrics)

            # Validate model
            self.validate(epoch)


    def validate(self, current_epoch: int):

        # Sets model into validation mode
        self.model.eval()

        # Initialize a defaultdict for accumulating metrics, if not already defined.
        running_metrics = collections.defaultdict(float)

        with torch.no_grad():
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("Validation", total=self.num_val_batches)
                # Stores last batch for reconstruction visualization
                last_batch = None
                for batch in self.val_loader:
                    metrics = self.validation_step(batch)
                    for key, value in metrics.items():
                        running_metrics[key] += value
                    progress.update(task, advance=1)
                    last_batch = batch


        # Average the metrics
        avg_metrics = {
            k: v / self.num_val_batches for k, v in running_metrics.items()
        }
        # Step scheduler if it exists
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_metrics['loss'])
        else:
            self.scheduler.step()
        self.training_logger.info("Scheduler updated using %s scheduler.", type(self.scheduler).__name__)
        self.training_logger.info("Average Validation Metrics: %s", avg_metrics)

        # Switch to validation phase
        self.experiment_logger.val()
        self.experiment_logger.log_model_metrics(avg_metrics)
        self.experiment_logger.end_epoch()


        # Check if the current validation loss is the best so far.
        current_val_loss = avg_metrics['loss']
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                # Clear the model directory so that only the best model remains.
                if os.path.exists(self.model_directory):
                    for file in os.listdir(self.model_directory):
                        file_path = os.path.join(self.model_directory, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    self.training_logger.info("Cleared model directory: %s", self.model_directory)
                else:
                    os.makedirs(self.model_directory, exist_ok=True)

                self.best_val_loss = current_val_loss
                self.best_epoch = current_epoch

                # Get the model name from the model's class name.
                model_name = self.model.model_id
                best_model_filename = f"{model_name}_best_model_epoch_{current_epoch}.pt"
                self.best_model_path = os.path.join(self.model_directory, best_model_filename)
                self.save_model(self.best_model_path)
                self.training_logger.info(
                    "Best model updated at epoch %d for model %s",
                    current_epoch, model_name, extra={'end_section': True}
                )

        self.post_validation(last_batch, current_epoch)

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
