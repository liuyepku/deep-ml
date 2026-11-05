import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = False)
    numOfBatches = 0
    batch_size = None
    for bx, by in dataloader:
        if batch_size == None:
            batch_size = tuple(bx.shape)
        numOfBatches+=1
    return numOfBatches, batch_size
