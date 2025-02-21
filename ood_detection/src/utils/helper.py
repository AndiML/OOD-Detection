import math
import torch

def compute_conv_output_size(size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    """
    Computes the output size of a convolutional layer along one dimension.
    """
    return math.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

