import logging
import torch

from ood_detection.src.trainer.base_trainer import BaseTrainer

class VAETrainer(BaseTrainer):
    """
    Trainer for Variational Autoencoders (VAEs).

    This trainer computes the VAE loss, which is the sum of the MSE-based reconstruction loss
    and the KL divergence between the approximate posterior and a standard Gaussian prior.

    """
    trainer_id = 'vae'

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
            model (nn.Module): The VAE model.
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
    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the VAE loss as the sum of the reconstruction loss and the KL divergence.

        Args:
            x (torch.Tensor): Original input data.
            x_hat (torch.Tensor): Reconstructed output from the model.
            mu (torch.ensor): Mean of the approximate posterior.
            logvar (torch.Tensor): Log variance of the approximate posterior.

        Returns:
            torch.Tensor: The total loss.
        """

        reconstruction_loss = torch.nn.functional.binary_cross_entropy(x_hat, x)
        kl_divergence_posterior_prior = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1).mean()

        return reconstruction_loss + kl_divergence_posterior_prior

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) ->  dict[str, float]:
        """
        Executes a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of training data.

        Returns:
            dict[str, float]: Returns the training loss over a batch.
        """
        inputs, _ = batch
        inputs = inputs.to(self.device)
        self.optimizer.zero_grad()
        outputs, mu, logvar = self.model(inputs)
        loss = self.loss_function(inputs, outputs, mu, logvar)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
        Executes a single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A batch of validation data, expected as (inputs, _).

        Returns:
           dict[str, float]: Returns the validation loss over a batch.
        """

        inputs, _ = batch
        inputs = inputs.to(self.device)
        outputs, mu, logvar = self.model(inputs)
        loss = self.loss_function(inputs, outputs, mu, logvar)
        return {'loss': loss.item()}


    def post_validation(self, last_batch: tuple[torch.Tensor, torch.Tensor], epoch: int) -> None:
        """
        Process the last validation batch to log reconstruction images.

        After the validation phase, this hook computes the reconstructed outputs for the
        last batch and logs both the original and reconstructed images via the experiment logger.

        Args:
            last_batch (tuple[torch.Tensor, torch.Tensor]): The last batch from validation.
            epoch (int): The current epoch number.
        """
        inputs, _ = last_batch
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs, _, _ = self.model(inputs)

        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu()

        self.experiment_logger.log_reconstruction_images(inputs, outputs, epoch=epoch)



    def compute_ood_score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative ELBO for each sample in x.
        The negative ELBO is used as the OOD score: lower ELBO (i.e., higher negative ELBO) indicates a potential anomaly.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: A 1D tensor of negative ELBO scores, one per sample.
        """
        self.model.eval()
        with torch.no_grad():
            outputs, mu, logvar = self.model(x)
            # Compute reconstruction loss per sample without summing across the batch.
            recon_loss = torch.nn.functional.binary_cross_entropy(outputs, x, reduction='none')
            recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)

            # Compute KL divergence per sample.
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # Negative ELBO per sample.
            neg_ellbo = -(recon_loss + kl_div)

        return neg_ellbo


