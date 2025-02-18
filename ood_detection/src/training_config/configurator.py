import torch

# ----------------------
# Optimizer Factories
# ----------------------

def get_optimizer(command_line_arguments, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Creates and returns an optimizer based on command-line arguments.

    Supported optimizer IDs: "adam", "sgd", "adamw", "rmsprop".

    command_line_arguments:
        command_line_arguments (Namespace): Command-line arguments; must contain attributes for the chosen optimizer.
        model (torch.nn.Module): The model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    optimizer_name = command_line_arguments.optimizer.lower()
    lr = command_line_arguments.learning_rate

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=command_line_arguments.weight_decay
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=command_line_arguments.momentum,
            weight_decay=command_line_arguments.weight_decay
        )

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=command_line_arguments.weight_decay
        )

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=command_line_arguments.momentum,
            weight_decay=command_line_arguments.weight_decay
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# ----------------------
# Scheduler Factories
# ----------------------

def get_scheduler(command_line_arguments, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Creates and returns a learning rate scheduler based on command-line arguments.

    Supported scheduler IDs: "steplr", "exponentiallr", "reducelronplateau",
                             "cosineannealinglr", "cosineannealingwarmrestarts".

    Args:
        command_line_arguments (Namespace): Command-line arguments; must contain attributes for the chosen scheduler.
        optimizer (torch.optim.Optimizer): The optimizer to be scheduled.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The scheduler instance, or None if not specified.
    """
    # Assume that the argument parser always sets an attribute named 'scheduler' (even if None)
    scheduler_name = command_line_arguments.scheduler
    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=command_line_arguments.step_size,
            gamma=command_line_arguments.gamma
        )

    elif scheduler_name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=command_line_arguments.gamma
        )

    elif scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=command_line_arguments.learning_rate_factor,
            patience=command_line_arguments.learning_rate_patience
        )

    elif scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=command_line_arguments.num_iteration_max,
            eta_min=command_line_arguments.minimum_learning_rate
        )

    elif scheduler_name == "cosineannealingwarmrestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=command_line_arguments.num_iteration_restart,
            T_mult=command_line_arguments.learning_increase_restart,
            eta_min=command_line_arguments.minimum_learning_rate
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")



