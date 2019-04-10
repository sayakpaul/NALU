import torch
from torch import nn
from NAC import NAC
from torch.nn import functional as F

class NALU(nn.Module):
    '''
    Class implementing Neural Arithmetic Logic Unit (NALU)
    with a small deviation from the original one described
    here: https://arxiv.org/abs/1808.00508
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = nn.Parameter(torch.Tensor(1, in_dim))
        nn.init.xavier_normal_(self.G)
        self.nac = NAC(self.in_dim, self.out_dim)
        self.eps = 1e-12

    def forward(self, x):
        g = torch.sigmoid(F.linear(x, self.G))
        y1 = g * self.nac(x)
        y2 = (1 - g) * torch.exp(self.nac(torch.log(torch.abs(x) + self.eps)))
        return y1 + y2