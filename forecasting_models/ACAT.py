import random
import numpy as np
import torch
import torch.nn as nn


class ACAT(nn.Module):

    def __init__(self, d_k, h, seed, device):

        super(ACAT, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device

        self.d_k = d_k
        self.filter_length = [3, 7, 9]
        self.conv_list_q = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       device=device) for f in self.filter_length])
        self.conv_list_k = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       device=device) for f in self.filter_length])
        self.norm = nn.BatchNorm1d(h * d_k).to(device)
        self.activation = nn.ELU()

    def forward(self, Q, K, V, mask=False):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        len_n_k = len(self.filter_length)

        Q_l = [self.activation(self.norm(self.conv_list_q[i](Q.reshape(b, h*d_k, l))))[:, :, :l]
               for i in range(len(self.filter_length))]
        K_l = [self.activation(self.norm(self.conv_list_k[i](K.reshape(b, h * d_k, l_k))))[:, :, :l_k]
               for i in range(len(self.filter_length))]
        Q_p = torch.cat(Q_l, dim=0).reshape(b, h, len_n_k, l, d_k)
        K_tmp = torch.cat(K_l, dim=0).reshape(b, h, len_n_k, l_k, d_k)

        m_f = max(self.filter_length)

        scores = torch.einsum('bhpqd,bhpkd->bhpqk', Q_p, K_tmp) / np.sqrt(self.d_k)

        if mask:

            mask = torch.tril(torch.ones(l, l_k)).to(torch.bool)
            mask = mask.unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).\
                repeat(1, h, 1, 1).unsqueeze(2).repeat(1, 1, len_n_k, 1, 1).to(self.device)
            scores.masked_fill_(mask, -1e10)

        attn = torch.softmax(scores, -1)
        attn, _ = torch.max(attn, dim=2)

        context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
        return context, attn