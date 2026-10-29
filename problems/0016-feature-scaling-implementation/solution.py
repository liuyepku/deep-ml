import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    mean = torch.mean(data_t, dim=0)
    std = torch.std(data_t, dim = 0, correction = 0)
    std = torch.where(std == 0, 1, std)
    std_data = (data_t - mean)/std

    minV = torch.min(data_t, dim=0).values 
    maxV = torch.max(data_t, dim = 0).values
    scale = maxV - minV
    scale = torch.where(scale == 0, 1, scale)
    scale_data = (data_t - minV)/scale
    return torch.round(std_data, decimals = 4), torch.round(scale_data, decimals = 4)
