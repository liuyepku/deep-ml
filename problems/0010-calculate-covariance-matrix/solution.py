import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    v_mean = torch.mean(v_t, dim=1, keepdims = True)
    v = v_t - v_mean
    cov = torch.matmul(v, v.T)/(v_t.shape[1]-1)
    return cov
