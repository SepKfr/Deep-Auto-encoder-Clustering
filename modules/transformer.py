import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer, TransformerDecoder, TransformerEncoder
from multi_head_attn import MultiHeadAttention


class Transformer(nn.Module):

    def __init__(self, d_model, nheads, num_layers, attn_type, seed):

        super(Transformer, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nheads)
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nheads)

        self.encoder_layer.self_attn = MultiHeadAttention(attn_type=attn_type,
                                                          d_model=d_model,
                                                          n_heads=nheads,
                                                          seed=seed)
        self.decoder_layer.self_attn = MultiHeadAttention(attn_type=attn_type,
                                                          d_model=d_model,
                                                          n_heads=nheads,
                                                          seed=seed)
        self.decoder_layer.multihead_attn = MultiHeadAttention(attn_type=attn_type,
                                                               d_model=d_model,
                                                               n_heads=nheads,
                                                               seed=seed)

        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, x_en, x_de):

        memory = self.encoder(x_en)
        output = self.decoder(x_de, memory)

        return output


if __name__ == "__main__":
    transformer = Transformer(d_model=8, nheads=1, num_layers=1, attn_type="ATA", seed=1234)

    x_en = torch.randn(16, 24, 8)
    x_de = torch.randn(16, 24, 8)

    output = transformer(x_en, x_de)
    print(output)