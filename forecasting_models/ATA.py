import torch
import torch.nn as nn
import numpy as np
import random

from seed_manager import set_seed


class ATA(nn.Module):
    def __init__(self, d_k, h, seed, device):

        super(ATA, self).__init__()

        set_seed(seed)

        self.d_k = d_k
        self.filter_length = [1, 3, 7, 9]
        self.device = device

        self.conv_list_k = nn.ModuleList([
            nn.Sequential(nn.Conv1d(
                in_channels=d_k*h, out_channels=d_k*h, kernel_size=f, padding=int((f-1)/2)),
                          nn.BatchNorm1d(d_k*h),
                          nn.ReLU()).to(device)
            for f in self.filter_length])

        self.conv_list_q = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h, kernel_size=f, padding=int((f-1)/2)),
                          nn.BatchNorm1d(d_k*h),
                          nn.ReLU()).to(device)
            for f in self.filter_length])

        self.proj_back_q = nn.Linear(d_k*len(self.filter_length), self.d_k)
        self.proj_back_k = nn.Linear(d_k*len(self.filter_length), self.d_k)

        self.factor = 1

    def forward(self, Q, K, V, mask=False):

        b, h, l, d_k = Q.shape

        l_k = K.shape[2]
        Q_l = []
        K_l = []

        Q = Q.reshape(b, -1, l)
        K = K.reshape(b, -1, l_k)

        [Q_l.append(self.conv_list_q[i](Q)) for i in range(len(self.filter_length))]
        [K_l.append(self.conv_list_k[i](K)) for i in range(len(self.filter_length))]

        Q_p = torch.cat(Q_l, dim=0).reshape(b, h, l * len(self.filter_length), -1)
        K_p = torch.cat(K_l, dim=0).reshape(b, h, l_k * len(self.filter_length), -1)

        Q_proj = Q_p.reshape(b, h, l, -1)
        Q, _ = torch.topk(Q_proj, dim=-1, k=1)

        K_proj = K_p.reshape(b, h, l_k, -1)
        K, _ = torch.topk(K_proj, dim=-1, k=1)

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        if mask:

            mask = torch.tril(torch.ones(l, l_k)).to(torch.bool)
            mask = mask.unsqueeze(0).repeat(b, 1, 1).unsqueeze(1).repeat(1, h, 1, 1).to(self.device)
            scores.masked_fill_(mask, -1e10)

        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
        return context, attn