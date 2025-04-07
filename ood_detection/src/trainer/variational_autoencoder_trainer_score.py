import torch
import torch.optim as optim

from ood_detection.src.trainer.variational_autoencoder_trainer import VAETrainer
from ood_detection.src.models.score_matcher import ScoreNet, denoising_score_matching_loss, langevin_dynamics, compute_outlier_score

class EnhancedVAETrainer(VAETrainer):
    """
    Enhanced VAE Trainer that uses a learned latent score function for improved OOD detection.

    This trainer extends the standard VAETrainer by:
      - Training a score network on the latent representations obtained from the VAE.
      - Refining latent codes using Langevin dynamics based on the learned score function.
      - Overriding the out-of-distribution (OOD) score computation to use the enhanced score.

    Inherits from:
        VAETrainer: The base trainer which computes OOD scores using negative ELBO.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the EnhancedVAETrainer.

        In addition to the standard VAETrainer initialization, it initializes a placeholder for the score network.

        Args:
            *args: Variable length argument list passed to the VAETrainer.
            **kwargs: Arbitrary keyword arguments passed to the VAETrainer.
        """
        super().__init__(*args, **kwargs)
        self.score_net = None

    def train_score_network(self, data_loader, noise_std=0.03, epochs=10, lr=1e-3, batch_size=64):
        """
        Trains the score network using denoising score matching on latent codes extracted from the VAE.

        This method first extracts latent representations (z) from the provided data loader using the VAE's encoder.
        It then trains a ScoreNet to predict the negative gradient of the log density of the latent distribution.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for extracting latent codes.
            noise_std (float): Standard deviation of the Gaussian noise used in the loss function.
            epochs (int): Number of training epochs for the score network.
            lr (float): Learning rate for the score network optimizer.
            batch_size (int): Batch size to use during training of the score network.

        Returns:
            ScoreNet: The trained score network.
        """
        # Extract latent codes from the VAE using the provided data loader.
        self.model.eval()
        latent_codes = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                # Obtain latent representation via the encoder (using reparameterization)
                _, mu, logvar = self.model(inputs)
                z = self.model.reparameterize(mu, logvar)
                latent_codes.append(z)
        latent_codes = torch.cat(latent_codes, dim=0)

        # Initialize the score network.
        latent_dim = latent_codes.size(1)
        self.score_net = ScoreNet(latent_dim).to(self.device)
        optimizer = optim.Adam(self.score_net.parameters(), lr=lr)

        # Create a DataLoader for the latent codes.
        latent_dataset = torch.utils.data.TensorDataset(latent_codes)
        latent_loader = torch.utils.data.DataLoader(latent_dataset, batch_size=batch_size, shuffle=True)

        # Train the score network.
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch_latents,) in latent_loader:
                batch_latents = batch_latents.to(self.device)
                optimizer.zero_grad()
                loss = denoising_score_matching_loss(self.score_net, batch_latents, noise_std)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.training_logger.info("ScoreNet Training - Epoch %d: Loss %.4f", epoch, epoch_loss / len(latent_loader))

        return self.score_net

    def compute_ood_score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes enhanced OOD scores for a batch of inputs using the trained score network.

        This method overrides the base class's OOD score computation (which uses negative ELBO).
        The steps are:
          1. Obtain the latent representations (z) from the VAE's encoder.
          2. Refine these latent codes via Langevin dynamics guided by the score network.
          3. Compute the outlier score as the L2 norm of the score network output on the refined codes.

        Args:
            x (torch.Tensor): Input batch of data.

        Returns:
            torch.Tensor: A 1D tensor of enhanced OOD scores for each sample in the batch.

        Raises:
            ValueError: If the score network has not been trained before calling this method.
        """
        if self.score_net is None:
            raise ValueError("Score network is not trained. Call train_score_network() before computing OOD scores.")

        self.model.eval()
        with torch.no_grad():
            _, mu, logvar = self.model(x)
            z = self.model.reparameterize(mu, logvar)

        # Refine latent codes using Langevin dynamics.

        refined_z = langevin_dynamics(z, self.score_net, step_size=0.1, num_steps=50, noise_std=0.03)
        # Compute the outlier score based on the learned score (using the L2 norm).
        ood_scores = compute_outlier_score(self.score_net, refined_z)
        return ood_scores
