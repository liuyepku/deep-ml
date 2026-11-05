import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    h = torch.tensor(initial_hidden_state, dtype = torch.float64)
    wx = torch.tensor(Wx, dtype = torch.float64)
    wh = torch.tensor(Wh, dtype = torch.float64)
    b = torch.tensor(b, dtype = torch.float64)

    for x in input_sequence:
        x = torch.tensor(x, dtype = torch.float64)
        h = torch.tanh(wx @ x + wh @ h + b)
    return h




