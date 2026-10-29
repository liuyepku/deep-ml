import torch

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between two tensors.

    Parameters:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    return (torch.mean(torch.abs(y_pred.float() - y_true.float()))).item()