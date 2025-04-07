"""Represents a module that contains the descriptor for the command for training and evaluation a model on medical datasets."""

from argparse import ArgumentParser

from ood_detection.commands.base import BaseCommandDescriptor
from ood_detection.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID
from ood_detection.src.models import MODEL_IDS, DEFAULT_MODEL_ID
from ood_detection.src.training_config import OPTIMIZER_IDS, DEFAULT_OPTIMIZER_ID, SCHEDULER_IDS, DEFAULT_SCHEDULER_ID


class OODPipelineCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of the OOD Pipeline command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """
        return 'ood-pipeline'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """
        return 'Trains the specified model on in-distribution data and evaluates its ability to detect out-of-distribution (OOD) data.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'output_path',
            type=str,
            help='Path to the directory where experiment results will be saved.'
        )

        parser.add_argument(
            'dataset_path',
            type=str,
            help='Path to the directory where in-distribution and OOD datasets are retrieved or downloaded.'
        )

        parser.add_argument(
            '-i',
            '--in_dataset',
            type=str,
            default=DEFAULT_DATASET_ID,
            choices=DATASET_IDS,
            help='Name of the in-distribution dataset used in training.'
        )

        parser.add_argument(
            '-o',
            '--ood_datasets',
            type=str,
            nargs='+',
            default=[d for d in DATASET_IDS if d != DEFAULT_DATASET_ID],
            choices=DATASET_IDS,
            help='List of out-of-distribution datasets to use during experiments.'
        )

        parser.add_argument(
            '-P',
            '--partition_method',
            type=str,
            default='external',
            choices=['internal', 'external'],
            help='Method for partitioning the test set: external (default) or internal'
        )
        parser.add_argument(
            '-L',
            '--num_inliers',
            type=int,
            default=3,
            help='Number of inlier classes to use for internal partitioning. Only used if partition_method is "internal".'
        )

        parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=5,
            help="Number of training epochs."
        )

        parser.add_argument(
            '-b',
            '--batchsize',
            type=int,
            default=64,
            help="Batch size during training."
        )

        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.01,
            help='Learning rate used during training.'
        )

        parser.add_argument(
            '-m',
            '--momentum',
            type=float,
            default=0.9,
            help='Momentum for the optimizer.'
        )

        parser.add_argument(
            '-w',
            '--weight_decay',
            type=float,
            default=0.0005,
            help='Weight decay used in the optimizer.'
        )

        # Model arguments
        parser.add_argument(
            '-t',
            '--model_type',
            type=str,
            default=DEFAULT_MODEL_ID,
            choices=MODEL_IDS,
            help='Type of neural network architecture used for training.'
        )

        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="If set, CUDA is utilized for training."
        )

        # Optimizer arguments
        parser.add_argument(
            '-p',
            '--optimizer',
            type=str,
            default=DEFAULT_OPTIMIZER_ID,
            choices=OPTIMIZER_IDS,
            help="Type of optimizer to use."
        )

        # Scheduler arguments
        parser.add_argument(
            '-s',
            '--scheduler',
            type=str,
            default=DEFAULT_SCHEDULER_ID,
            choices=SCHEDULER_IDS,
            help="Type of learning rate scheduler to use."
        )

        # Scheduler-specific arguments
        parser.add_argument(
            '-S',
            '--step_size',
            type=int,
            default=10,
            help='Step size for StepLR scheduler (default: 10).'
        )

        parser.add_argument(
            '-G',
            '--gamma',
            type=float,
            default=0.1,
            help='Decay factor for StepLR or ExponentialLR scheduler (default: 0.1).'
        )

        parser.add_argument(
            '-F',
            '--learning_rate_factor',
            type=float,
            default=0.1,
            help='Factor by which the LR is reduced in ReduceLROnPlateau scheduler (default: 0.1).'
        )

        parser.add_argument(
            '-A',
            '--learning_rate_patience',
            type=int,
            default=5,
            help='Number of epochs with no improvement before reducing LR in ReduceLROnPlateau scheduler (default: 5).'
        )

        parser.add_argument(
            '-N',
            '--num_iteration_max',
            type=int,
            default=50,
            help='Maximum iterations for CosineAnnealingLR scheduler (default: 50).'
        )

        parser.add_argument(
            '-R',
            '--minimum_learning_rate',
            type=float,
            default=0.0,
            help='Minimum learning rate for cosine annealing schedulers (default: 0.0).'
        )

        parser.add_argument(
            '-I',
            '--learning_increase_restart',
            type=int,
            default=2,
            help='Factor for increasing the restart period in CosineAnnealingWarmRestarts scheduler (default: 2).'
        )

        parser.add_argument(
            '-C',
            '--num_iteration_restart',
            type=int,
            default=10,
            help='Number of iterations for a restart in CosineAnnealingWarmRestarts scheduler (default: 10).'
        )


        # Model-specific arguments
        parser.add_argument(
            '-d',
            '--latent_dim',
            type=int,
            default=100,
            help='Dimensionality of the latent representation in reconstruction based models.'
        )

        parser.add_argument(
            '-f',
            '--min_feature_size',
            type=int,
            default=None,
            help='Minimum feature size (optional).'
        )
        parser.add_argument(
            '-B',
            '--base_channels',
            type=int,
            default=None,
            help='Minimum base channels.'
        )

        parser.add_argument(
             '-n',
            '--noise_std',
            type=float,
            default=0.0,
            help='Standard deviation of noise added to the input image.'
        )

         # Enhanced OOD detection arguments for score matching.
        parser.add_argument(
            '--enhanced_ood',
            action='store_true',
            help='If set, use the enhanced OOD detection method with score matching and Langevin dynamics.'
        )

        parser.add_argument(
            '--score_epochs',
            type=int,
            default=10,
            help='Number of training epochs for the latent score network.'
        )

        parser.add_argument(
            '--score_lr',
            type=float,
            default=1e-3,
            help='Learning rate for training the latent score network.'
        )

        parser.add_argument(
            '--score_noise_std',
            type=float,
            default=0.03,
            help='Standard deviation of Gaussian noise for training the latent score network.'
        )

        parser.add_argument(
            '--ld_step_size',
            type=float,
            default=0.1,
            help='Step size for Langevin dynamics in enhanced OOD detection.'
        )

        parser.add_argument(
            '--ld_num_steps',
            type=int,
            default=50,
            help='Number of steps for Langevin dynamics in enhanced OOD detection.'
        )

