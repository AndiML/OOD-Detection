
# List of supported optimizer IDs
OPTIMIZER_IDS = [
    "adam",
    "sgd",
    "adamw",
    "rmsprop"
]

DEFAULT_OPTIMIZER_ID = "adam"

# List of supported scheduler IDs
SCHEDULER_IDS = [
    "steplr",
    "exponentiallr",
    "reducelronplateau",
    "cosineannealinglr",
    "cosineannealingwarmrestarts"
]

DEFAULT_SCHEDULER_ID = "steplr"


__all__ = [
    "create_optimizer",
    "OPTIMIZER_IDS",
    "DEFAULT_OPTIMIZER_ID",
    "create_scheduler",
    "SCHEDULER_IDS",
    "DEFAULT_SCHEDULER_ID",
]
