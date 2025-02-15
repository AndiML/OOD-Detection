import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional

from ood_detection.src.models.base_model import BaseModel


def compute_conv_output_size(size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    """
    Computes the output size of a convolutional layer along one dimension.
    """
    return math.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class AutoDynamicAutoencoder(BaseModel):
    """
    An autoencoder that automatically builds its convolutional layers given:
      - The known input channel dimension.
      - The known input spatial size.

    The encoder will apply a sequence of convolutions (with kernel_size=3, stride=2, padding=1)
    until the spatial size is reduced to a specified minimum. The number of output channels is increased
    at each layer (by doubling in this example). The decoder mirrors the encoder structure using transposed convolutions.

    Optionally, Gaussian noise can be injected into the input during training.
    """

    model_id = 'ae'

    def __init__(self,
        input_channels: int,
        input_size: int,
        latent_dim: int,
        min_feature_size: int =  None,
        base_channels: int = None,
        noise_std: float = 0.0) -> None:
        """
        Initializes the autoencoder.

        Args:
            input_channels (int): Number of channels in the input image.
            input_size (int): Spatial size of the input image (assuming square images.
            latent_dim (int): The dimensionality of the latent representation.
            min_feature_size (int, optional): The minimum spatial dimension allowed in the encoder.
                                              If None, defaults are chosen based on input_size.
            base_channels (int, optional): The number of output channels for the first convolution.
                                           If None, defaults are chosen based on input_size.
            noise_std (float): Standard deviation of Gaussian noise to inject during training (default: 0.0).
        """
        super(AutoDynamicAutoencoder, self).__init__()
        self.noise_std = noise_std
        # Set default parameters based on the input size if not provided.
        if input_size <= 50:
            min_feature_size = 4 if min_feature_size is None else min_feature_size
            base_channels = 32 if base_channels is None else base_channels
        elif input_size <= 100:
            min_feature_size = 8 if min_feature_size is None else min_feature_size
            base_channels = 64 if base_channels is None else base_channels
        else:  # input_size > 100
            min_feature_size = 16 if min_feature_size is None else min_feature_size
            base_channels = 128 if base_channels is None else base_channels

        self.encoder_convs = nn.ModuleList()
        # Save pairs (in_channels, out_channels) for later mirror building.
        encoder_channels = []

        current_channels = input_channels
        current_size = input_size
        out_channels = base_channels

        # Build encoder layers until the spatial size reaches the minimum.
        while current_size > min_feature_size:
            convolution_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.encoder_convs.append(convolution_layer)
            encoder_channels.append((current_channels, out_channels))
            # Update for next layer.
            current_channels = out_channels
            current_size = compute_conv_output_size(current_size, kernel_size=3, stride=2, padding=1)
            out_channels *= 2  # Example rule: double the channels each time.

        self.final_channels = current_channels
        self.feature_map_size = current_size  # Final spatial size from encoder.

        # Fully connected layers to bridge between conv feature maps and latent space.
        self.fc_enc = nn.Linear(self.final_channels * self.feature_map_size * self.feature_map_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.final_channels * self.feature_map_size * self.feature_map_size)

        # Build the decoder as a mirror of the encoder.
        self.decoder_deconvs = nn.ModuleList()
        # Reverse the channel pairs.
        reversed_channels = encoder_channels[::-1]


        for idx, (in_ch, out_ch) in enumerate(reversed_channels):
            # Use output_padding=0 for the first layer to correctly map 4->7 instead of 4->8.
            op = 0 if idx == 0 else 1
            deconvolution_layer = nn.ConvTranspose2d(
                in_channels=out_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=op
            )
            self.decoder_deconvs.append(deconvolution_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_channels, input_size, input_size).

        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Inject noise if desired.
        if self.noise_std > 0.0 and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Encoder
        for conv in self.encoder_convs:
            x = F.relu(conv(x))
        # Flatten the output of the last conv layer.
        x = x.view(x.size(0), -1)
        latent = self.fc_enc(x)

        # Decoder
        x = self.fc_dec(latent)
        x = x.view(x.size(0), self.final_channels, self.feature_map_size, self.feature_map_size)
        for deconv in self.decoder_deconvs:
            x = F.relu(deconv(x))

        reconstruction = torch.nn.functional.sigmoid(x)
        return reconstruction
