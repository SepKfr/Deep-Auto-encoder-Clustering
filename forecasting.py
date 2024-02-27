import torch
from torch import nn
import random
import numpy as np
from modules.transformer import Transformer

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Forecasting(nn.Module):
    def __init__(self, input_size, output_size,
                 d_model, nheads, num_layers,
                 attn_type, seed, device,
                 pred_len, batch_size):

        super(Forecasting, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)

        self.forecasting_model = Transformer(d_model, d_model, nheads=nheads, num_layers=num_layers,
                                             attn_type=attn_type, seed=seed, device=device)
        self.fc_dec = nn.Linear(d_model, output_size)

        self.d_model = d_model
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.device = device

    def forward(self, x, y=None,):

        loss = 0

        x = self.embedding(x)

        forecast_enc, forecast_dec = self.forecasting_model(x)

        forecast_out = self.fc_dec(forecast_dec)[:, -self.pred_len:, :]

        if y is not None:

            loss = nn.MSELoss()(y, forecast_out)

        return forecast_out, loss
