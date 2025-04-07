import os
import logging
from datetime import datetime
from argparse import Namespace


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
            experiment_logger=experiment_logger,
            enhanced_ood=command_line_arguments.enhanced_ood
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
