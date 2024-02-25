import numpy as np
import torch.nn as nn
import torch
from torch.nn import Linear
from GMM import GmmFull, GmmDiagonal
from modules.transformer import Transformer
torch.autograd.set_detect_anomaly(True)


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, attn_type, seed, device):
        super(Encoder, self).__init__()

        self.encoder = Transformer(input_size=d_model, d_model=d_model*2,
                                   nheads=n_heads, num_layers=num_layers,
                                   attn_type=attn_type, seed=seed, device=device)
        self.d_model = d_model

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):

        enc_output, dec_output = self.encoder(x)
        output = torch.cat([enc_output, dec_output], dim=1)
        mean = output[:, :, :self.d_model]
        log_var = output[:, :, -self.d_model:]
        f_output = self.reparameterize(mean, log_var)
        return f_output, mean, log_var


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size, n_unique,
                 d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size,
                 num_clusters):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.enc_embedding = nn.Linear(input_size, d_model)

        self.encoder = Encoder(d_model=d_model, n_heads=nheads, num_layers=num_layers,
                               attn_type=attn_type, seed=seed, device=device)

        # to learn change points
        self.enc_proj_down = nn.Linear(d_model, n_unique+1)

        self.centroids = nn.Parameter(torch.randn(num_clusters, d_model))

        self.fc_dec = Linear(d_model, output_size)

        self.w = nn.Parameter(torch.randn(2))

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model

    def forward(self, x, y=None):

        loss_total = 0

        x_silver_standard = x[:, :, -1].to(torch.long)

        x = self.enc_embedding(x)

        f_output, mean, log_var = self.encoder(x)

        output_cp = self.enc_proj_down(f_output)

        dists = torch.einsum('bsd, cd -> bsc', f_output, self.centroids) / np.sqrt(self.d_model)
        forecast_out = torch.einsum('bsc, cd-> bsd', dists, self.centroids)

        final_out = self.fc_dec(forecast_out)[:, -self.pred_len:, :]

        if y is not None:

            output_cp = output_cp.permute(0, 2, 1)
            loss_cp = nn.CrossEntropyLoss()(output_cp, x_silver_standard)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            cp_loss_total = loss_cp + torch.clip(self.w[0], min=0, max=0.005) * kl_divergence

            mse_loss = nn.MSELoss()(y, final_out)

            loss_total = mse_loss + torch.clip(self.w[1], min=0, max=0.005) * cp_loss_total

        return final_out, loss_total
