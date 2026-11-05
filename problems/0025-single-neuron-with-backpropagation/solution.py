import torch
import torch.nn as nn
import numpy as np


def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    X = torch.tensor(features, dtype = torch.float64)
    y = torch.tensor(labels, dtype = torch.float64)
    w = torch.tensor(initial_weights, dtype = torch.float64, requires_grad = True)
    b = torch.tensor(initial_bias, dtype = torch.float64, requires_grad = True)
    loss_values = []
    for i in range(epochs):
        prob = torch.sigmoid(X @ w + b)
        
        loss = ((y - prob)**2).mean()
        loss_values.append(loss.item())

        loss.backward()

        with torch.no_grad():
            w -= learning_rate* w.grad
            b -= learning_rate* b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    return (np.round(w.detach().numpy(), 4).tolist(), np.round(b.detach().numpy(), 4).tolist(), loss_values)
