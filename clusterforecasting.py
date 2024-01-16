import torch.nn as nn
import torch
from torch.nn import Linear
from modules.transformer import Transformer
from zeroInflatedGaussian import ZeroInflatedGaussian


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size,
                 num_clusters, d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, seq_length,
                 num_seg, batch_size):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.embedding_1 = nn.Linear(input_size, d_model, bias=False)
        self.embedding_2 = nn.Linear(input_size, d_model)

        self.zero_inflated = ZeroInflatedGaussian(num_clusters, batch_size,
                                                  num_seg, seq_length,
                                                  d_model)

        self.forecasting_model = Transformer(d_model, d_model, nheads=nheads, num_layers=num_layers,
                                             attn_type=attn_type, seed=seed)
        self.fc_dec = Linear(d_model, output_size)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                 nn.ReLU(),
                                 nn.Linear(d_model*4, d_model))

        self.lam = nn.Parameter(torch.randn(1), requires_grad=True)

        self.pred_len = pred_len
        self.num_clusters = num_clusters
        self.nheads = nheads
        self.num_seg = num_seg
        self.batch_size = batch_size
        self.seq_len = seq_length
        self.d_model = d_model

    def forward(self, x_1, x_2, y=None):

        tot_loss = 0
        x_1 = self.embedding_2(x_1)
        x = x_2.permute(0, 2, 1, 3)
        x = self.embedding_1(x)

        sample, cluster_loss = self.zero_inflated(x)

        sample = sample.permute(1, 2, 3, 4, 0)

        # Get cluster assignment indices along the 'S' dimension
        cluster_indices = torch.argmax(sample, dim=-1)

        # Create a one-hot encoding tensor based on the cluster indices
        one_hot_encoding = torch.zeros_like(sample)
        one_hot_encoding.scatter_(-1, cluster_indices.unsqueeze(-1), 1)

        # Sum along the 'S' and 'D' dimension using broadcasting
        sum_by_cluster = (sample * one_hot_encoding).mean(dim=-1)

        tensor = (sum_by_cluster + x).mean(1)

        tensor = tensor.reshape(self.batch_size, -1, self.d_model)

        x = torch.split(tensor, split_size_or_sections=int(tensor.shape[1]/2), dim=1)

        x_enc = x[0]
        x_dec = x[1]

        x_app = torch.zeros((self.batch_size, self.pred_len, self.d_model), device=self.device)
        x_dec = torch.cat([x_dec, x_app], dim=1)

        forecast_enc, forecast_dec = self.forecasting_model(x_enc, x_dec)
        output_dec = x_1[:, -self.pred_len:, :] + self.ffn(forecast_dec[:, -self.pred_len:, :])

        forecast_out = self.fc_dec(output_dec)

        if y is not None:

            loss = nn.MSELoss()(y, forecast_out)
            tot_loss = loss + torch.clamp(self.lam, min=0, max=0.001)

        return forecast_out, tot_loss
