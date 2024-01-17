import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer, TransformerDecoder, TransformerEncoder, Linear
from modules.multi_head_attn import MultiHeadAttention


class Transformer(nn.Module):

    def __init__(self, input_size, d_model, nheads, num_layers, attn_type, seed):

        super(Transformer, self).__init__()

        self.enc_embedding = Linear(input_size, d_model)
        self.dec_embedding = Linear(input_size, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nheads)
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nheads)

        self.encoder_layer.self_attn = MultiHeadAttention(attn_type=attn_type,
                                                          d_model=d_model,
                                                          n_heads=nheads,
                                                          seed=seed,
                                                          batch_first=False)
        self.decoder_layer.self_attn = MultiHeadAttention(attn_type=attn_type,
                                                          d_model=d_model,
                                                          n_heads=nheads,
                                                          seed=seed,
                                                          batch_first=False)
        self.decoder_layer.multihead_attn = MultiHeadAttention(attn_type=attn_type,
                                                               d_model=d_model,
                                                               n_heads=nheads,
                                                               seed=seed,
                                                               batch_first=False)

        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, x_en, x_de):

        x_en = self.enc_embedding(x_en)
        x_de = self.enc_embedding(x_de)
        memory = self.encoder(x_en)
        output = self.decoder(x_de, memory)

        return memory, output


if __name__ == "__main__":

    transformer = Transformer(input_size=8, d_model=8, nheads=1, num_layers=1, attn_type="ATA", seed=1234)

    x_en = torch.randn(16, 24, 8)
    x_de = torch.randn(16, 24, 8)

    _, output = transformer(x_en, x_de)
    print(output)