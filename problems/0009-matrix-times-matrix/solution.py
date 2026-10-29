import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a = torch.tensor(a, dtype = torch.float32)
    b = torch.tensor(b, dtype = torch.float32)
    if a.shape[1]!=b.shape[0]:
        return -1
    
    return torch.matmul(a, b)
