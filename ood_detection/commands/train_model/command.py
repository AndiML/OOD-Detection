"""Represents a module that contains the command for the download of the respective dataset."""

import csv
import logging
from argparse import Namespace
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from ood_detection.commands.base import BaseCommand
from ood_detection.src.datasets.dataset import Dataset
from ood_detection.src.models.variational_autoencoder import  AutoDynamicVariationalAutoencoder
from ood_detection.src.models.autoencoder import  AutoDynamicAutoencoder
import torch.nn as nn


class TrainModelCommand(BaseCommand):
    """Represents a command that represents the train model command."""

    def __init__(self) -> None:
        """Initializes a new TrainModel instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Downloads the specified dataset
        self.logger.info("Downloading %s Dataset for OOD Detection", command_line_arguments.dataset.upper(), extra={'start_section': True} )

        # Device configuration.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dataset loaders.
        # Assume Dataset.create() returns an object with methods to get training, validation, and test loaders.
        dataset = Dataset.create(command_line_arguments.dataset, command_line_arguments.dataset_path)
        train_loader = dataset.get_training_data_loader(batch_size=64, shuffle_samples=True)
        validation_loader = dataset.get_validation_data_loader(batch_size=64)
        test_loader = dataset.get_test_data_loader(batch_size=1)
        input_shape = dataset.sample_shape


        # Create the autoencoder model.
        model = AutoDynamicVariationalAutoencoder(
            input_channels=input_shape[0],
            input_size=input_shape[1],
            latent_dim=8 # For MedMNIST, 16 is a reasonable starting point.
        ).to(device)
        print(model)

        # Define the loss function.
        def loss_function(x, x_hat, mean, log_var):
            # Reconstruction loss using Mean Squared Error.
            reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
            # Kullback-Leibler divergence.
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return reconstruction_loss + KLD

        criterion = loss_function

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # If you have pretrained weights, you can load them.
        # model.load_state_dict(torch.load("auto_dynamic_vae.pth"))

        # Set up a cosine annealing scheduler.
        num_epochs = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        print_interval = 10

        # Lists to store metrics for each epoch.
        train_losses = []
        val_losses = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            # Progress bar for training batches.
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

            for batch_idx, (inputs, _) in progress_bar:
                inputs = inputs.to(device)

                optimizer.zero_grad()
                outputs, mean, log_var = model(inputs)  # Forward pass.
                loss = criterion(inputs, outputs, mean, log_var)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # Update the progress bar with current loss.
                if batch_idx % print_interval == 0:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")

            # Evaluate on the validation set.
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in validation_loader:
                    inputs = inputs.to(device)
                    outputs, mean, log_var = model(inputs)
                    loss = criterion(inputs, outputs, mean, log_var)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(validation_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch [{epoch}] Validation Loss: {avg_val_loss:.4f}")

            # Step the scheduler.
            scheduler.step()

        # Save the trained model.
        torch.save(model.state_dict(), "test.pth")
        print("Model saved as test.pth")

        # Save the training and validation metrics to a CSV file.
        metrics_file = "metrics.csv"
        with open(metrics_file, mode='w', newline='') as csv_file:
            fieldnames = ['epoch', 'train_loss', 'val_loss']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(num_epochs):
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_losses[epoch],
                    'val_loss': val_losses[epoch]
                })
        print(f"Metrics saved to {metrics_file}")

        # ------------------------------
        # Variational Inference Section
        # ------------------------------
        # Create a folder to save inference images.
        inference_dir = "inference_images"
        os.makedirs(inference_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            print("Performing variational inference on test data...")
            for batch_idx, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                reconstruction, mu, logvar = model(inputs)

                # Print latent variables.
                print(f"Sample {batch_idx}:")
                print("Latent mean (mu):", mu.cpu().numpy())
                print("Latent log variance (logvar):", logvar.cpu().numpy())

                # Sample a latent vector using reparameterization.
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                print("Sampled latent vector (z):", z.cpu().numpy())

                # Visualize and save original and reconstructed images.
                # Assuming the image tensor shape is (batch, channels, height, width).
                original = inputs[0].cpu().numpy().transpose(1, 2, 0)
                reconstructed = reconstruction[0].cpu().numpy().transpose(1, 2, 0)

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(original.squeeze(), cmap='gray' if input_shape[0] == 1 else None)
                plt.title("Original")
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(reconstructed.squeeze(), cmap='gray' if input_shape[0] == 1 else None)
                plt.title("Reconstructed")
                plt.axis('off')

                # Save the figure.
                image_path = os.path.join(inference_dir, f"sample_{batch_idx}.png")
                plt.savefig(image_path)
                plt.close()
                print(f"Saved inference image: {image_path}")

                # Optionally, process a few samples (here we break after 3 samples).
                if batch_idx >= 2:
                    break
