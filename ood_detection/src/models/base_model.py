"""A module that contains the abstract base class for models."""

from abc import ABC, abstractmethod

import torch

class BaseModel(ABC, torch.nn.Module):
    """Represents the abstract base class for all models."""

     # Registry to automatically register subclasses.
    _registry: dict[str, type["BaseModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically register subclasses that define a model_id attribute.
        if hasattr(cls, "model_id"):
            model_id = getattr(cls, "model_id")
            cls._registry[model_id] = cls

    def __init__(self, task_type: str) -> None:
        """_Initializes a BaseModel instance."""

        self.task_type = task_type
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.
        Returns:
            torch.Tensor: Returns the outputs of the model.
        """
        raise NotImplementedError
