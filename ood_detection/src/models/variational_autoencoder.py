import torch
import torch.nn as nn
import torch.nn.functional as F

from ood_detection.src.models.base_model import BaseModel
from ood_detection.src.utils.helper import compute_conv_output_size

class AutoDynamicVariationalAutoencoder(BaseModel):
    """
    A variational autoencoder (VAE) that automatically builds its convolutional layers
    based on the known input channels and spatial size.

    The encoder applies a series of convolutional layers until the spatial size is reduced to a
    specified minimum. The number of channels is increased at each layer (doubling, in this example).
    The decoder mirrors the encoder using transposed convolutions. The VAE produces both a mean and
    log variance for the latent space and uses the reparameterization trick to sample a latent vector.

    Optionally, Gaussian noise can be injected into the input during training.
    """

    model_id = 'vae'
    task_type = 'reconstruction'

    def __init__(self,
        input_channels: int,
        input_size: int,
        latent_dim: int,
        min_feature_size: int,
        base_channels: int,
        noise_std: float,
    ) -> None:
        """
        Initializes the AutoDynamicVariationalAutoencoder instance.

        Args:
            input_channels (int): Number of channels in the input image.
            input_size (int): Spatial size of the input image.
            latent_dim (int): Dimensionality of the latent space.
            min_feature_size (int): Minimum spatial dimension allowed in the encoder.
            base_channels (int): Number of output channels for the first convolution.
            noise_std (float): Standard deviation for Gaussian noise injected during training.
        """
        super(AutoDynamicVariationalAutoencoder, self).__init__(task_type='reconstruction')
        self.noise_std = noise_std

        # Set default parameters based on the input size if not provided.
        if input_size <= 50:
            min_feature_size = 4 if min_feature_size is None else min_feature_size
            base_channels = 32 if base_channels is None else base_channels
        elif input_size <= 100:
            min_feature_size = 8 if min_feature_size is None else min_feature_size
            base_channels = 64 if base_channels is None else base_channels
        else:
            min_feature_size = 16 if min_feature_size is None else min_feature_size
            base_channels = 128 if base_channels is None else base_channels


        self.encoder_convs = nn.ModuleList()
        # Keep track of (in_channels, out_channels) pairs for mirroring in the decoder.
        self.encoder_channels = []

        current_channels = input_channels
        current_size = input_size
        out_channels = base_channels

        # Build the encoder layers until the spatial size is reduced to the minimum.
        while current_size > min_feature_size:
            # Use a strided convolution for downsampling.
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.encoder_convs.append(conv_layer)
            current_size = compute_conv_output_size(current_size, kernel_size=3, stride=2, padding=1)
            self.encoder_channels.append((current_channels, out_channels))
            current_channels = out_channels
            out_channels *= 2  # Double channels at each layer.

        self.final_channels = current_channels
        self.feature_map_size = current_size  # Final spatial dimension from the encoder.

        # Fully connected layers for latent space mapping.
        self.fc_mu = nn.Linear(self.final_channels * self.feature_map_size * self.feature_map_size, latent_dim)
        self.fc_logvar = nn.Linear(self.final_channels * self.feature_map_size * self.feature_map_size, latent_dim)

        # Fully connected layer to map the latent vector back to the flattened feature maps.
        self.fc_dec = nn.Linear(latent_dim, self.final_channels * self.feature_map_size * self.feature_map_size)

        # Build the decoder as a mirror of the encoder.
        self.decoder_deconvs = nn.ModuleList()
        # Reverse the list of channel pairs for decoder construction.
        reversed_channels = self.encoder_channels[::-1]
        # Otherwise, use transposed convolutions to upsample.
        for idx, (in_ch, out_ch) in enumerate(reversed_channels):
            # Adjust output_padding for the first layer if needed.
            op = 0 if idx == 0 else 1
            deconv_layer = nn.ConvTranspose2d(
                in_channels=out_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=op
            )
            self.decoder_deconvs.append(deconv_layer)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.

        Returns:
            torch.Tensor: A sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_channels, input_size, input_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - The reconstructed image.
                - The mean (mu) of the latent distribution.
                - The log variance (logvar) of the latent distribution.
        """
        # Optionally inject Gaussian noise during training.
        if self.noise_std > 0.0 and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Encoder: pass through convolutional layers.
        for layer in self.encoder_convs:
            x =  F.relu(layer(x))
        x = x.view(x.size(0), -1)

        # Obtain latent distribution parameters.
        mu = self.fc_mu(x)
        # Predict log_var for numeric stability since the network can output an arbitrary number
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder: map latent vector back to feature maps.
        x = self.fc_dec(z)
        x = x.view(x.size(0), self.final_channels, self.feature_map_size, self.feature_map_size)
        for layer in self.decoder_deconvs:
            x = F.relu(layer(x))
        # Use sigmoid to constrain the output in [0, 1] (assuming normalized images).
        reconstruction = F.sigmoid(x)
        return reconstruction, mu, logvar

    def decode(self, z: torch.Tensor = None, mu: torch.Tensor = None, logvar: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes a latent vector z into an image reconstruction.

        Args:
        z (torch.Tensor): A latent vector of shape (batch_size, latent_dim)
        mu (torch.Tensor): The mean from which z will be sampled using the reparameterization trick.
        logvar (torch.Tensor): The variance from which z will be sampled using the reparameterization trick.

        If neither z nor both mu and logvar are provided, the function will raise an error.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        if z is None:
            if mu is not None and logvar is not None:
                z = self.reparameterize(mu, logvar)
            else:
                raise ValueError("Provide either a latent vector 'z' or both 'mu' and 'logvar'.")

        # Map latent vector back to feature maps.
        x = self.fc_dec(z)
        x = x.view(x.size(0), self.final_channels, self.feature_map_size, self.feature_map_size)

        # Pass through the decoder deconvolutional layers.
        for deconv in self.decoder_deconvs:
            x = F.relu(deconv(x))

        reconstruction = F.sigmoid(x)

        return reconstruction

