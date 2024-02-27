import numpy as np
import torch
import torch.nn as nn
import random

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class BasicAttn(nn.Module):

    def __init__(self, d_k):

        super(BasicAttn, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V):

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
        return context, attn