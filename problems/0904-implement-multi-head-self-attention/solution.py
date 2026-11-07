import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, d_model, bias = False)
        self.Wk = nn.Linear(d_model, d_model, bias = False)
        self.Wv = nn.Linear(d_model, d_model, bias = False)
        self.Wo = nn.Linear(d_model, d_model, bias = False)


    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B, T, D = x.shape
        d_heads = D // self.num_heads

        assert(D % self.num_heads == 0)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # B, T, num_heads, d_heads -> B, num_heads, T, d_heads
        Q = Q.reshape(B, T, self.num_heads, d_heads).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, d_heads).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, d_heads).transpose(1, 2)

        # B, num_heads, T, T
        scores = Q @ K.transpose(-2, -1) / (d_heads ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim = -1)
        # B, num_heads, T, d_heads
        outputs = weights @ V
        outputs = outputs.transpose(1, 2).reshape(B, T, D)

        return self.Wo(outputs)




