import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    mat = torch.tensor(matrix, dtype = torch.float64)
    vals = torch.linalg.eigvals(mat)
    return torch.sort(vals.real, descending = True).values
    
