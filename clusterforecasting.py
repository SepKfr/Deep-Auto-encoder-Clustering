import torch.nn as nn
import torch
from torch.nn import Linear

from GMM import GmmFull, GmmDiagonal
from modules.transformer import Transformer


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size,
                 d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size,
                 gmm_model):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.embedding = nn.Linear(input_size, d_model)
        self.gmm_model = gmm_model

        if self.gmm_model is not None:

            self.gmm_embedding = nn.Linear(self.gmm_model.num_dims, d_model)

        self.forecasting_model = Transformer(d_model, d_model, nheads=nheads, num_layers=num_layers,
                                             attn_type=attn_type, seed=seed, device=self.device)
        self.fc_dec = Linear(d_model, output_size)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                 nn.ReLU(),
                                 nn.Linear(d_model*4, d_model))
        self.norm = nn.LayerNorm(d_model)

        self.lam = nn.Parameter(torch.randn(1), requires_grad=True)

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model

    def forward(self, x, y=None):

        mse_loss = 0

        if self.gmm_model is not None:
            _, sample = self.gmm_model(x)
            sample = self.gmm_embedding(sample)
            x = self.embedding(x)
            x = x + sample
        else:
            x = self.embedding(x)

        x = torch.split(x, split_size_or_sections=int(x.shape[1]/2), dim=1)

        x_enc = x[0]
        x_dec = x[1]

        x_app = torch.zeros((self.batch_size, self.pred_len, self.d_model), device=self.device)
        x_dec = torch.cat([x_dec, x_app], dim=1)

        forecast_enc, forecast_dec = self.forecasting_model(x_enc, x_dec)

        forecast_out = self.fc_dec(forecast_dec)[:, -self.pred_len:, :]

        if y is not None:

            mse_loss = nn.MSELoss()(y, forecast_out)

        return forecast_out, mse_loss
