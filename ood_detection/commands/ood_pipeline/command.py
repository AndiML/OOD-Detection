import logging
from datetime import datetime
from argparse import Namespace
import os


from ood_detection.commands.base import BaseCommand
from ood_detection.src.datasets.dataset import Dataset
from ood_detection.src.experiments.tracker import ExperimentLogger
from ood_detection.src.datasets.data_partitioner import DataPartitioner
from ood_detection.src.models.model_generator import create_model
from ood_detection.src.datasets.dataset_config import report_dataset_configuration
from ood_detection.src.trainer.trainer_generator import create_trainer_from_model
from ood_detection.src.training_config.configurator import get_optimizer, get_scheduler


class OODPipelineCommand(BaseCommand):
    """Represents a command that represents the  OODPipeline command."""

    def __init__(self) -> None:
        """Initializes a new OODPipeline instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

         # Log dataset download and configuration.
        self.logger.info("Downloading in-distribution dataset: %s",command_line_arguments.in_dataset.upper())
        in_dataset_instance = Dataset.create(command_line_arguments.in_dataset, command_line_arguments.dataset_path)

        # Configure device.
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info("Using device: %s for training on dataset: %s", device.upper(), command_line_arguments.in_dataset
        )

        # Define training and model checkpoint paths.
        training_dir = command_line_arguments.output_path
        checkpoint_dir = os.path.join(training_dir, "model-checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info("Created checkpoint directory: %s", checkpoint_dir)

        # Initialize dataset and dataloaders.
        self.logger.info("Retrieving Training and Validation Set for: %s", command_line_arguments.in_dataset)
        train_loader = in_dataset_instance.get_training_data_loader(batch_size=command_line_arguments.batchsize, shuffle_samples=True)
        valid_loader = in_dataset_instance.get_validation_data_loader(batch_size=command_line_arguments.batchsize)

        # Create the model.
        self.logger.info("Creating model of type: %s", command_line_arguments.model_type)
        model = create_model(in_dataset_instance, command_line_arguments)
        self.logger.info("Model created with task type: %s", model.task_type)

        # Initialize experiment logger.
        self.logger.info("Initializing Experiment Logger")
        experiment_logger = ExperimentLogger(
            output_path=training_dir,
            task_type=model.task_type,
            logger=self.logger
        )
        experiment_logger.display_hyperparamter_for_in_data_training(command_line_arguments)

        # Save hyperparameters.
        hyperparameters = vars(command_line_arguments)
        hyperparameters['start_date_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        experiment_logger.save_hyperparameters(hyperparameters)

        # Retrieve optimizer and scheduler based on command-line arguments.
        self.logger.info("Configuring optimizer: %s", command_line_arguments.optimizer)
        optimizer = get_optimizer(command_line_arguments, model)
        self.logger.info("Optimizer configured successfully.")

        self.logger.info("Configuring scheduler: %s", command_line_arguments.scheduler)
        scheduler = get_scheduler(command_line_arguments, optimizer)
        if scheduler is not None:
            self.logger.info("Scheduler configured successfully.")
        else:
            self.logger.info("No scheduler configured.")

        # Create trainer using model's ID to select the appropriate trainer class.
        self.logger.info("Instantiating trainer for model ID: %s", model.model_id)
        trainer = create_trainer_from_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=valid_loader,
            device=device,
            num_epochs=command_line_arguments.epochs,
            experiment_path=training_dir,
            scheduler=scheduler,
            training_logger=self.logger,
            experiment_logger=experiment_logger
        )
        trainer.train()

        if command_line_arguments.partition_method.lower() == "internal":
            self.logger.info(
                f"Partitioning method is '{command_line_arguments.partition_method}'. "
                "Since the partitioning is done internally on the in-distribution dataset, no external check is needed."
            )
            partitioner = DataPartitioner(in_dataset=in_dataset_instance)
            partitioner.partition(
                partition_method="internal",
                num_inliers=command_line_arguments.num_inliers
            )
            test_loader = partitioner.get_dataloader(command_line_arguments.batchsize)
             # Compute anomaly scores for the internal partition.
            all_scores, all_labels = trainer.compute_ood_scores(test_loader)
            experiment_logger.log_ood_metrics(all_labels, all_scores, partition_strategy="internal", partition_info=command_line_arguments.num_inliers)

        else:
            report_dataset_configuration(
                in_data_class_instance=in_dataset_instance,
                ood_dataset_ids=command_line_arguments.ood_datasets,
                logger=self.logger
            )
            for ood_id in command_line_arguments.ood_datasets:
                partitioner = DataPartitioner(in_dataset=in_dataset_instance)
                partitioner.partition(
                    partition_method="external",
                    ood_dataset_id=ood_id,
                    dataset_path=command_line_arguments.dataset_path
                )
                test_loader = partitioner.get_dataloader(command_line_arguments.batchsize)

                 # Compute anomaly scores for the internal partition.
                all_scores, all_labels = trainer.compute_ood_scores(test_loader)
                experiment_logger.log_ood_metrics(all_labels, all_scores, partition_strategy="external", partition_info=ood_id)

            exit()

        # # Save the training and validation metrics to a CSV file.
        # with open(metrics_filename, mode='w', newline='') as csv_file:
        #     fieldnames = ['epoch', 'train_loss', 'val_loss']
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for epoch in range(num_epochs):
        #         writer.writerow({
        #             'epoch': epoch + 1,
        #             'train_loss': train_losses[epoch],
        #             'val_loss': val_losses[epoch]
        #         })
        # print(f"Metrics saved to {metrics_filename}")

        # # ------------------------------
        # # Variational Inference Section
        # # ------------------------------
        # model.eval()
        # with torch.no_grad():
        #     print("Performing variational inference on test data...")
        #     for batch_idx, (inputs, _) in enumerate(test_loader):
        #         inputs = inputs.to(device)
        #         reconstruction, mu, logvar = model(inputs)

        #         # Print latent variables.
        #         print(f"Sample {batch_idx}:")
        #         print("Latent mean (mu):", mu.cpu().numpy())
        #         print("Latent log variance (logvar):", logvar.cpu().numpy())

        #         # Sample a latent vector using reparameterization.
        #         std = torch.exp(0.5 * logvar)
        #         eps = torch.randn_like(std)
        #         z = mu + eps * std
        #         print("Sampled latent vector (z):", z.cpu().numpy())

        #         # Visualize and save original and reconstructed images.
        #         # Assuming the image tensor shape is (batch, channels, height, width).
        #         original = inputs[0].cpu().numpy().transpose(1, 2, 0)
        #         reconstructed = reconstruction[0].cpu().numpy().transpose(1, 2, 0)

        #         plt.figure(figsize=(8, 4))
        #         plt.subplot(1, 2, 1)
        #         plt.imshow(original.squeeze(), cmap='gray' if input_shape[0] == 1 else None)
        #         plt.title("Original")
        #         plt.axis('off')
        #         plt.subplot(1, 2, 2)
        #         plt.imshow(reconstructed.squeeze(), cmap='gray' if input_shape[0] == 1 else None)
        #         plt.title("Reconstructed")
        #         plt.axis('off')

        #         # Save the inference image with an experiment-specific filename.
        #         image_path = os.path.join(inference_dir, f"{experiment_name}_sample_{batch_idx}.png")
        #         plt.savefig(image_path)
        #         plt.close()
        #         print(f"Saved inference image: {image_path}")

        #         # Process only a few samples (e.g., break after 3 samples).
        #         if batch_idx >= 2:
        #             break
