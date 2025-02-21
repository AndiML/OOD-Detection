import logging
import torch
import torch.utils
from ood_detection.src.trainer.base_trainer import BaseTrainer

class AETrainer(BaseTrainer):
    """
    Trainer for a basic Autoencoder (AE).
    """
    trainer_id = 'ae'

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
        experiment_logger = None
    ) -> None:

        """
        Args:
            model (nn.Module): The AutoEncoder model.
            optimizer (optim.Optimizer): Optimizer for model parameters.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device for computation.
            num_epochs (int): Number of training epochs.
            training_logger (logging.Logger): The logger for the training process.
        """

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            experiment_path=experiment_path,
            training_logger=training_logger,
            scheduler=scheduler,
            experiment_logger=experiment_logger
        )

        # Use MSE loss for reconstruction.
        self.criterion = torch.nn.BCELoss()

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Executes a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of training data.

        Returns:
            float: Returns the training loss over a batch.
        """

        inputs, _ = batch
        inputs = inputs.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, inputs)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Executes a single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of validation data, expected as (inputs, _).

        Returns:
           float: Returns the validation loss over a batch.
        """

        inputs, _ = batch
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, inputs)

        return {'loss': loss.item()}


    def compute_ood_score_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        For an autoencoder, anomaly scores are computed as the reconstruction error.
        Returns:
            A 1D tensor of anomaly scores (one per sample).
        """
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(batch.to(self.device))
            error = self.criterion(reconstruction, batch.to(self.device))
            # Sum error over all non-batch dimensions
            error = error.view(error.size(0), -1).sum(dim=1)

        return error

