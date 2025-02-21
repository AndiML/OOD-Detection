import torch

class ChannelAdjustmentTransform:
    """
    Adjusts the number of channels in a tensor.
    If the tensor has fewer channels than desired, it repeats the channels.
    If it has more channels than desired, it selects the first channel(s).
    """
    def __init__(self, desired_channels: int):
        self.desired_channels = desired_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        current_channels = x.shape[0]
        if current_channels == self.desired_channels:
            return x
        elif current_channels < self.desired_channels:
            # Repeat the channel(s) until the desired number is reached.
            repeat_factor = self.desired_channels // current_channels
            x = x.repeat(repeat_factor, 1, 1)
            return x[:self.desired_channels]
        else:
            # More channels than needed: select the first desired_channels.
            return x[:self.desired_channels]

class OODTransformer:
    """
    A combined transformer that adjusts the input channels and transforms the label.
    The mode flag determines the label: 'known' becomes 1 and 'unknown' becomes 0.
    """
    def __init__(self, desired_channels: int, mode: str):
        if mode not in ['known', 'unknown']:
            raise ValueError("Mode must be either 'known' or 'unknown'")
        self.desired_channels = desired_channels
        self.mode = mode
        self.channel_adjuster = ChannelAdjustmentTransform(desired_channels)

    def __call__(self, x, y):
        # Adjust channels for the input tensor.
        x = self.channel_adjuster(x)
        # Set label based on mode.
        label = 1 if self.mode == 'known' else 0
        return x, label

class TransformWrapperOOD(torch.utils.data.Dataset):
    """
    A dataset wrapper that applies a combined transformer (for both inputs and labels)
    to each sample.
    """
    def __init__(self, base_dataset: torch.utils.data.Dataset, transformer):
        self.base_dataset = base_dataset
        self.transformer = transformer

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        return self.transformer(x, y)
