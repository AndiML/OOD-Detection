"""Represents a module containing the creation of the model for training on In-Data."""
from argparse import Namespace
import torch

from ood_detection.src.models.base_model import BaseModel

def create_model(data_class_instance:torch.utils.data.Dataset, command_line_arguments: Namespace) -> BaseModel:
    """
    Creates a model instance based on command-line arguments and dataset information.

    Args:
        data_class_instance (torch.utils.data.Dataset): An instance of a dataset (or configuration) class that provides input_channels and input_size.
        command_line_arguments (Namespace): Parsed command-line arguments that include `model_type` and any model-specific parameters.

    Returns:
        An instance of a model that is a subclass of BaseModel.

    Raises:
        ValueError: If no model is registered with the specified model_type.
    """
    # Get the model class from the registry using the model_type argument.
    model_instance = BaseModel._registry.get(command_line_arguments.model_type)
    if model_instance is None:
        raise ValueError(f"Unknown model type: {command_line_arguments.model_type}. "
                         f"Available types: {list(BaseModel._registry.keys())}")

    # Gather any additional model-specific parameters from the command-line arguments.
    model_params = {}
    sample_shape = data_class_instance.sample_shape
    model_params["input_channels"] = sample_shape[0]
    model_params["input_size"] = sample_shape[1]
    model_params["latent_dim"] = command_line_arguments.latent_dim
    model_params["min_feature_size"] = command_line_arguments.min_feature_size
    model_params["base_channels"] = command_line_arguments.base_channels
    model_params["noise_std"] = command_line_arguments.noise_std

    # Instantiate the model.
    model = model_instance(**model_params)

    return model

def create_model(data_class_instance: torch.utils.data.Dataset, command_line_arguments: Namespace) -> BaseModel:
    """
    Creates a model instance based on command-line arguments and dataset information.
    Supports both reconstruction models and classification models.

    Args:
        data_class_instance (torch.utils.data.Dataset): An instance of a dataset (or configuration) class that provides input_channels and input_size.
        command_line_arguments (Namespace): Parsed command-line arguments that include the type of model and any model-specific parameters.

    Returns:
        An instance of a model that is a subclass of BaseModel.

    Raises:
        ValueError: If no model is registered with the specified model_type.

    """
    # Look up the model class from the registry.
    model_class = BaseModel._registry.get(command_line_arguments.model_type)
    if model_class is None:
        raise ValueError(
            f"Unknown model type: {command_line_arguments.model_type}. "
            f"Available types: {list(BaseModel._registry.keys())}"
        )

    sample_shape = data_class_instance.sample_shape

    # Common parameters
    model_params = {"input_channels": sample_shape[0], "input_size": sample_shape[1]}

    task_type = model_class.task_type
    if task_type == "reconstruction":
        model_params.update({
            "latent_dim": command_line_arguments.latent_dim,
            "min_feature_size": command_line_arguments.min_feature_size,
            "base_channels": command_line_arguments.base_channels,
            "noise_std": command_line_arguments.noise_std,
        })
    elif task_type == "classification":
        model_params.update({
            "num_classes": command_line_arguments.num_classes,
        })
    else:
        raise ValueError(f"Unknown task_type: {command_line_arguments.task_type}")

    # Instantiate the model.
    model = model_class(**model_params)
    return model
