import torch
from torch import nn

from modules.transformer import Transformer


class Forecasting(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 d_model,
                 nheads,
                 num_layers,
                 attn_type,
                 seed,
                 device,
                 pred_len,
                 batch_size
                 ):
        super(Forecasting, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)

        self.forecasting_model = Transformer(d_model, d_model, nheads=nheads, num_layers=num_layers,
                                             attn_type=attn_type, seed=seed)
        self.fc_dec = nn.Linear(d_model, output_size)

        self.d_model = d_model
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.device = device

    def forward(self, x_1, y=None):

        loss = 0
        x_1 = self.embedding(x_1)

        x = torch.split(x_1, split_size_or_sections=int(x_1.shape[1] / 2), dim=1)

        x_enc = x[0]
        x_dec = x[1]

        x_app = torch.zeros((self.batch_size, self.pred_len, self.d_model), device=self.device)
        x_dec = torch.cat([x_dec, x_app], dim=1)

        _, output = self.forecasting_model(x_enc, x_dec)

        final_output = self.fc_dec(output[:, -self.pred_len:, :])

        if y is not None:

            loss = nn.MSELoss()(y, final_output)

        return final_output, loss
