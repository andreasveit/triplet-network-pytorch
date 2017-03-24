import torch
#from torch.autograd import Variable
#import numpy as np
import torch.nn as nn

# distance
# TODO: replace this function once pairwise distance is available as part of Pytorch
def pairwise_distance(x1, x2, p=2, eps=1e-6):
    """
    Computes the batchwise pairwise distance between vectors v1,v2:
        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}
        Args:
            x (Tensor): input tensor containing the two input batches
            p (real): the norm degree. Default: 2
        Shape:
            - Input: :math:`(N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.pairwise_distance(input1, input2, p=2)
        >>> output.backward()
    """
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).sum(dim=1)
    return torch.pow(out, 1. / p)

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z