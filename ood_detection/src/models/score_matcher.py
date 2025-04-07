import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    """
    A simple neural network that learns the score function of the latent distribution.

    This network takes a latent vector as input and outputs a vector of the same size,
    which is interpreted as the gradient (score) of the log-density with respect to the latent vector.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        hidden_dim (int, optional): Number of neurons in the hidden layer. Defaults to 128.
    """
    def __init__(self, latent_dim, hidden_dim=128):
        super(ScoreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        """
        Forward pass through the score network.

        Args:
            z (torch.Tensor): Input latent vector of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output score vector of shape (batch_size, latent_dim).
        """
        return self.net(z)

def denoising_score_matching_loss(score_net, z, noise_std):
    """
    Computes the denoising score matching loss.

    This function adds Gaussian noise to the latent codes and trains the score network
    to predict the gradient of the log density of the latent distribution.

    Args:
        score_net (ScoreNet): The score network to be trained.
        z (torch.Tensor): The original latent codes, shape (batch_size, latent_dim).
        noise_std (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        torch.Tensor: The mean squared error loss between the predicted score and the target score.
    """
    # Add Gaussian noise to the latent vector
    noisy_z = z + torch.randn_like(z) * noise_std

    # Predict the score for the noisy latent code
    score = score_net(noisy_z)

    # Target score is the negative gradient of the Gaussian noise model
    target_score = -(noisy_z - z) / (noise_std ** 2)

    # Compute the mean squared error loss between the predicted and target scores
    loss = torch.mean((score - target_score) ** 2)
    return loss

def langevin_dynamics(z_init, score_net, step_size=0.1, num_steps=50, noise_std=0.03):
    """
    Samples/refines latent codes using Langevin dynamics guided by the score network.

    This function iteratively updates the latent codes in the direction of the score (gradient)
    and adds noise at each step to ensure proper exploration of the latent space.

    Args:
        z_init (torch.Tensor): Initial latent codes, shape (batch_size, latent_dim).
        score_net (ScoreNet): The trained score network.
        step_size (float, optional): Step size for the update. Defaults to 0.1.
        num_steps (int, optional): Number of Langevin dynamics steps. Defaults to 50.
        noise_std (float, optional): Standard deviation of the noise to add in each step. Defaults to 0.03.

    Returns:
        torch.Tensor: Refined latent codes after applying Langevin dynamics, shape (batch_size, latent_dim).
    """
    z = z_init.clone().detach()
    for _ in range(num_steps):
        z.requires_grad = True
        score = score_net(z)
        z = z + 0.5 * step_size * score
        z = z + torch.randn_like(z) * (step_size ** 0.5)
        z = z.detach()
    return z

def compute_outlier_score(score_net, z):
    """
    Computes an outlier score for each latent code based on the score network.

    The outlier score is calculated as the L2 norm of the score vector.
    Higher scores indicate a higher likelihood that the sample is out-of-distribution.

    Args:
        score_net (ScoreNet): The trained score network.
        z (torch.Tensor): Latent codes for which to compute the score, shape (batch_size, latent_dim).

    Returns:
        torch.Tensor: A 1D tensor containing the outlier score for each sample.
    """
    score = score_net(z)

    # Calculate the L2 norm of the score as the outlier score
    outlier_score = torch.norm(score, dim=1)
    return outlier_score
