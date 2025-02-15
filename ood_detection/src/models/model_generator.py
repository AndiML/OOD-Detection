"""Represents a module containing the creation of the global model for the federated learning process."""
import sys
import torch

from ood_detection.src.models.vgg import Vgg11




def create_global_model(
        model_type: str,
        dataset_kind: str,
        data_class_instance, #Dataset,
        tensor_shape_for_flattening: tuple[int, ...],
        use_larger_model: bool = False
) -> torch.nn.Module:
    """Creates the global model for the federated learning process

    Args:
        model_type (str): The type of model architecture to be used.
        dataset_kind (str): The kind of dataset used in the federated learning process.
        data_class_instance (Dataset): The instance of the class containing the dataset.
        tensor_shape_for_flattening (tuple[int, ...]): The shape of single sample in the training to initialize the first linear layer of
            a simple feed forward neural network.
        use_larger_model (bool): Whether to use the larger VGG-11 model for training.

    Returns:
        torch.nn.Module: The global model for the federated learning process.
    """
    global_model: torch.nn.Module
    if model_type == 'cnn':
        # Convolutional neural network architecture
        if dataset_kind == 'mnist':
            img_size = tensor_shape_for_flattening
            input_dimension = 1
            for size_per_dimension in img_size:
                input_dimension *= size_per_dimension
        elif dataset_kind == 'cifar10' or dataset_kind == 'cifar100-super' or dataset_kind == 'cinic10':
            if use_larger_model:
                global_model = Vgg11(input_shape=data_class_instance.sample_shape, number_of_classes=data_class_instance.number_of_classes)
            else:
                global_model = CNNCifar10(
                    number_of_channels=data_class_instance.sample_shape[0],
                    output_classes=data_class_instance.number_of_classes
                )
        elif dataset_kind == 'cifar100' or dataset_kind == 'imagenet':
            if use_larger_model:
                global_model = Vgg11(input_shape=data_class_instance.sample_shape, number_of_classes=data_class_instance.number_of_classes)
            else:
                global_model = CNNCifar10(
                    number_of_channels=data_class_instance.sample_shape[0],
                    output_classes=data_class_instance.number_of_classes
                )
            # global_model = create_resnet50_cifar100(num_classes=data_class_instance.number_of_classes)
            # CNNCifar100(input_shape=data_class_instance.sample_shape, output_classes=data_class_instance.number_of_classes)
        else:
            exit('Model not supported.')

    elif model_type == 'mlp':
        # Multi-layer perceptron
        img_size = tensor_shape_for_flattening
        input_dimension = 1
        for size_per_dimension in img_size:
            input_dimension *= size_per_dimension
        global_model = None
    elif model_type == 'classifier':
        global_model = None

    elif model_type == 'encoder':
        global_model = None
    else:
        sys.exit('Architecture not supported')

    return global_model
