import torch
import math

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    result = 1 / (1 + math.exp(-z))
    return result
