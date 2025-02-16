"""Represents a module that contains experiment management utilities."""

import os
import csv
import logging
from datetime import datetime
from argparse import Namespace
from typing import Any, Optional, Dict

import torchvision
import yaml  # type: ignore
import torch
from torch.utils.tensorboard.writer import SummaryWriter

class ExperimentLogger:
    """
    A versatile experiment logger that supports:
      - Saving hyperparameters (YAML).
      - Logging per-epoch training metrics to a CSV file.
      - Logging metrics to TensorBoard (if enabled).
      - Automatic handling of model-specific metrics based on a model tag and task type.
      - (Optional) Logging OOD metrics using pytorch_ood.

    The logger expects a task type which defines the set of expected metrics:
      - For 'reconstruction': expects ['Reconstruction_Error'].
      - For 'classification': expects ['Accuracy', 'Loss'].
    """

    def __init__(
        self,
        output_path: str,
        task_type: str,
        logger: logging.Logger,
        use_tensorboard: bool = False,
        ood_logging: bool = True
    ) -> None:
        """
        Initializes the experiment logger.

        Args:
            output_path (str): Directory where logs will be saved.
            task_type (str): Type of task (e.g., 'reconstruction', 'classification').
            logger(logging.Logger): Logger is provided to log metrics obtained during the training process directly to the command line.
            use_tensorboard (bool): If True, TensorBoard logging is enabled.
            ood_logging (bool): If True, enables OOD metrics logging.
        """
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.task_type = task_type
        self.logger = logger
        self.use_tensorboard = use_tensorboard
        self.ood_logging = ood_logging
        self.phase = "train"

        # Define expected metrics based on the task type
        if self.task_type == "reconstruction":
            self.expected_metrics_keys = ["loss"]
        elif self.task_type == "classification":
            self.expected_metrics_keys = ["loss", "accuracy"]
        else:
            self.expected_metrics_keys = []

        # Set up CSV logging.
        self.metrics_file_path = os.path.join(self.output_path, 'metrics.csv')
        self.metrics_file = open(self.metrics_file_path, 'w', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.metrics_file)
        self.csv_header = ['Epoch', 'Timestamp']
        self.has_written_csv_header = False

        # Initializes TensorBoard logging if enabled.
        if self.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=self.output_path)
        else:
            self.tensorboard_writer = None

        self.current_epoch: Optional[int] = None
        self.current_epoch_metrics: Dict[str, Any] = {}

        # Initializes OOD metrics logging.
        if self.ood_logging:
            try:
                from pytorch_ood.utils import OODMetrics
                self.ood_metrics_calculator = OODMetrics()
            except ImportError:
                print("pytorch_ood is not installed. OOD logging will be disabled.")
                self.ood_logging = False
                self.ood_metrics_calculator = None
        else:
            self.ood_metrics_calculator = None

    def display_hyperparamter_for_in_data_training(self, command_line_arguments: Namespace) -> None:
        """Displays the parameters that are used during the training.

        Args:
            command_line_arguments (Namespace): The command line arguments that are used in the training process over the in distribution data.

        """
        self.logger.info('\nExperimental details:')
        self.logger.info(f'Model                             : {command_line_arguments.model_type.upper()}')
        self.logger.info(f'Optimizer                         : {command_line_arguments.optimizer.upper()}')
        self.logger.info(f'Learning Rate                     : {command_line_arguments.learning_rate}')
        self.logger.info(f'Epochs                            : {command_line_arguments.epochs}\n')
        self.logger.info('Training Parameters')
        self.logger.info(f'Batch size                        : {command_line_arguments.batchsize}')
        self.logger.info(f' GPU enabled:                     : {command_line_arguments.use_gpu}\n')

    def save_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Saves hyperparameters to a YAML file.

        Args:
            hyperparameters (Dict[str, Any]): A dictionary of hyperparameter names and values.
        """
        hp_path = os.path.join(self.output_path, 'hyperparameters.yaml')
        with open(hp_path, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparameters, f)

    def train(self) -> None:
        """
        Sets the current logging phase to training.
        """
        self.phase = "train"
        self.logger.info("Switched logger phase to Train.")

    def val(self) -> None:
        """
        Sets the current logging phase to validation.
        """
        self.phase = "val"
        self.logger.info("Switched logger phase to Validation.")

    def begin_epoch(self, epoch: int) -> None:
        """
        Initializes logging for a new epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.current_epoch = epoch
        self.current_epoch_metrics = {
            'Epoch': epoch,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def add_metric(self, name: str, title: str, value: float) -> None:
        """
        Logs a single metric.

        Args:
            name (str): The name of the metric. This is used as the header in the resulting CSV file.
            title (str): The human-readable name of the metric. This is used as the title in TensorBoard.
            value (float): The value of the metric.
        """

        # Checks if the metric is new, if so, then it is added to the header (unless the header has already been written to file, in that case the
        # number of columns cannot be changed)
        if not self.has_written_csv_header and name not in self.csv_header:
            self.csv_header.append(name)

        # Writes the metric to the TensorBoard log file
        if self.tensorboard_writer is not None and self.current_epoch is not None:
            self.tensorboard_writer.add_scalar(title, value, self.current_epoch)

        # Caches the metric, it will be written to disk once the metrics are committed
        self.current_epoch_metrics[name] = value

    def log_model_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Logs a set of metrics based on the task type and current phase.
        Expects a dictionary where the keys correspond to expected metric names.

        For example:
          - For a reconstruction model: logger.log_model_metrics({'loss': loss_value})
          - For a classification model: logger.log_model_metrics({'accuracy': accuracy_value, 'loss': loss_value})

        Args:
            metrics (Dict[str, float]): Dictionary containing metric values.
        """
        # Check for missing expected metrics.
        missing = [key for key in self.expected_metrics_keys if key not in metrics]
        if missing:
            self.logger.warning(
                f"Missing expected metric(s): {', '.join(missing)}. Provided: {list(metrics.keys())}"
            )

        phase_prefix = self.phase.capitalize()
        for key in self.expected_metrics_keys:
            if key in metrics:
                value = metrics[key]
                metric_name = f"{phase_prefix}_{key}"
                title = f"{phase_prefix}/{key}"
                self.add_metric(name=metric_name, title=title, value=value)

    def end_epoch(self) -> None:
        """
        Writes out the current epoch's metrics to the CSV file.
        """
        if not self.has_written_csv_header:
            self.csv_writer.writerow(self.csv_header)
            self.has_written_csv_header = True

        row = [self.current_epoch_metrics.get(col, None) for col in self.csv_header]
        self.csv_writer.writerow(row)
        self.metrics_file.flush()

    def log_ood_metrics(self, in_scores: torch.Tensor, out_scores: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        Computes and logs OOD metrics if enabled.

        This method should be called after training (or at evaluation time) to log OOD detection performance.

        Args:
            in_scores (torch.Tensor): In-distribution scores.
            out_scores (torch.Tensor): Out-of-distribution scores.

        Returns:
            Optional[Dict[str, float]]: A dictionary containing computed OOD metrics, or None if OOD logging is disabled.
        """
        if not self.ood_logging or self.ood_metrics_calculator is None:
            self.logger.info("OOD logging is not enabled.")
            return None

        metrics = self.ood_metrics_calculator(in_scores, out_scores)
        global_step = self.current_epoch if self.current_epoch is not None else 0
        for metric_name, value in metrics.items():
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar(f'OOD/{metric_name}', value, global_step)

        # Write OOD metrics to a separate CSV file.
        ood_csv_path = os.path.join(self.output_path, 'ood_metrics.csv')
        write_header = not os.path.isfile(ood_csv_path)
        with open(ood_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Timestamp'] + list(metrics.keys()))
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + list(metrics.values()))
        return metrics

    def log_reconstruction_images(self, original: torch.Tensor, reconstructed: torch.Tensor, epoch: int) -> None:
        """
        Logs a batch of original and reconstructed images for reconstruction-based models.

        If TensorBoard is enabled, the images are logged via TensorBoard.
        Otherwise, the images are saved as files in an 'images' subfolder within the output directory.

        Args:
            original (torch.Tensor): A batch of original images (shape: [B, C, H, W]).
            reconstructed (torch.Tensor): A batch of reconstructed images (shape: [B, C, H, W]).
            epoch (int): The current epoch number, used for logging.
        """
        # Create image grids for original and reconstructed images.
        orig_grid = torchvision.utils.make_grid(original)
        recon_grid = torchvision.utils.make_grid(reconstructed)

        if self.tensorboard_writer is not None:
            # Log images using TensorBoard.
            self.tensorboard_writer.add_image('Original', orig_grid, epoch)
            self.tensorboard_writer.add_image('Reconstructed', recon_grid, epoch)
        else:
            # Save images to disk.
            images_dir = os.path.join(self.output_path, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Convert tensor grids to PIL images.
            to_pil = torchvision.transforms.ToPILImage()
            orig_img = to_pil(orig_grid)
            recon_img = to_pil(recon_grid)

            # Save the images with epoch-based filenames.
            orig_img.save(os.path.join(images_dir, f'epoch_{epoch}_original.png'))
            recon_img.save(os.path.join(images_dir, f'epoch_{epoch}_reconstructed.png'))


    def set_state(self, state: dict) -> None:
        self.phase = state.get('phase', 'train')
        self.current_epoch = state.get('current_epoch', None)
        self.current_epoch_metrics = state.get('current_epoch_metrics', {})

        # Re-open the CSV file for appending metrics
        self.metrics_file = open(self.metrics_file_path, 'a', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.metrics_file)

        # Optionally, reinitialize TensorBoard writer if needed.
        if self.use_tensorboard:
            # Using the same log_dir ensures that new events are added to the existing ones.
            self.tensorboard_writer = SummaryWriter(log_dir=self.output_path)

    def close(self) -> None:
        """
        Closes file handles and the TensorBoard writer.
        """
        if not self.metrics_file.closed:
            self.metrics_file.close()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

