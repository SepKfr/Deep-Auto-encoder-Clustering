import random

import numpy as np
import torch
import torch.nn as nn
from forecasting_models.ACAT import ACAT
from forecasting_models.ATA import ATA
from forecasting_models.ConvAttn import ConvAttn
from forecasting_models.BasicAttn import BasicAttn
from forecasting_models.Informer import ProbAttention
from forecasting_models.Autoformer import AutoCorrelation


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, attn_type, seed, device):

        super(MultiHeadAttention, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = device

        d_k = int(d_model / n_heads)
        d_v = d_k
        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.seed = seed

    def forward(self, Q, K, V):

        batch_size = Q.shape[0]

        q_s = self.WQ(Q).reshape(batch_size, self.n_heads, -1, self.d_k)
        k_s = self.WK(K).reshape(batch_size, self.n_heads, -1, self.d_k)
        v_s = self.WV(V).reshape(batch_size, self.n_heads, -1, self.d_k)

        # ATA forecasting model

        if self.attn_type == "ATA":
            context, attn = ATA(d_k=self.d_k, h=self.n_heads, seed=self.seed, device=self.device)(
            Q=q_s, K=k_s, V=v_s)

        elif self.attn_type == "ACAT":
            context, attn = ACAT(d_k=self.d_k, h=self.n_heads, seed=self.seed)(
            Q=q_s, K=k_s, V=v_s)

        # Autoformer forecasting model

        elif self.attn_type == "autoformer":
            context, attn = AutoCorrelation(seed=self.seed)(q_s.transpose(1, 2),
                                                            k_s.transpose(1, 2),
                                                            v_s.transpose(1, 2))

        # CNN-trans forecasting model

        elif self.attn_type == "conv_attn":
            context, attn = ConvAttn(d_k=self.d_k, seed=self.seed, kernel=9, h=self.n_heads)(
            Q=q_s, K=k_s, V=v_s)

        # Informer forecasting model

        elif self.attn_type == "informer":
            context, attn = ProbAttention(mask_flag=False, seed=self.seed)(q_s, k_s, v_s)

        else:
            context, attn = BasicAttn(d_k=self.d_k)(Q=q_s, K=k_s, V=v_s)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        outputs = self.fc(context)
        return outputs