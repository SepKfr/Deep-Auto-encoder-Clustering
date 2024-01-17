import torch.nn as nn
import torch
from torch.nn import Linear
from modules.transformer import Transformer
from GMM import GMM


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size,
                 num_clusters, d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.embedding_1 = nn.Linear(input_size, d_model, bias=False)
        self.embedding_2 = nn.Linear(input_size, d_model)

        self.gmm = GMM(num_clusters, d_model)

        self.forecasting_model = Transformer(d_model, d_model, nheads=nheads, num_layers=num_layers,
                                             attn_type=attn_type, seed=seed, device=self.device)
        self.fc_dec = Linear(d_model, output_size)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                 nn.ReLU(),
                                 nn.Linear(d_model*4, d_model))

        self.lam = nn.Parameter(torch.randn(1), requires_grad=True)

        self.pred_len = pred_len
        self.num_clusters = num_clusters
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model

    def forward(self, x, y=None):

        tot_loss = 0
        x_init = self.embedding_2(x)

        x = torch.split(x_init, split_size_or_sections=int(x_init.shape[1]/2), dim=1)

        x_enc = x[0]
        x_dec = x[1]

        x_enc_short, loss_enc = self.gmm(x_enc)
        x_dec_short, loss_dec = self.gmm(x_dec)

        x_app = torch.zeros((self.batch_size, self.pred_len, self.d_model), device=self.device)
        x_dec = torch.cat([x_dec, x_app], dim=1)

        forecast_enc, forecast_dec = self.forecasting_model(x_enc, x_enc_short, x_dec, x_dec_short)

        output_dec = x_init[:, -self.pred_len:, :] + self.ffn(forecast_dec[:, -self.pred_len:, :])

        forecast_out = self.fc_dec(output_dec)

        if y is not None:

            loss = nn.MSELoss()(y, forecast_out)
            if self.training:
                tot_loss = loss + torch.clamp(self.lam, min=0, max=0.001) * (loss_enc + loss_dec)
            else:
                tot_loss = loss

        return forecast_out, tot_loss
