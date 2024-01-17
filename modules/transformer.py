import torch
import torch.nn as nn
from modules.multi_head_attn import MultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.device = device

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(self.device)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, attn_type, seed):

        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type, seed=seed)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type, seed=seed)

        self.pos_ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                     nn.ReLU(),
                                     nn.Linear(d_model*4, d_model))

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

    def forward(self, q, k, c, enc_outputs):

        out = self.dec_self_attn(q, k, c)
        out = self.layer_norm_1(q + out)
        out2 = self.dec_enc_attn(out, enc_outputs, enc_outputs)
        out2 = self.layer_norm_2(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm_3(out2 + out3)
        return out3


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, d_model, device):
        super(Decoder, self).__init__()

        self.pos_emb = PositionalEncoding(d_model=d_model, device=device)
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, q, k, v, enc_outputs):

        q = self.pos_emb(q)
        for i in range(self.num_layers):
            q = self.decoder_layers[i](q, k, v, enc_outputs)

        return q


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, attn_type, seed):

        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type, seed=seed)

        self.pos_ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                     nn.ReLU(),
                                     nn.Linear(d_model*4, d_model))

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, q, k, c):

        out = self.enc_self_attn(q, k, c)
        out2 = self.layer_norm_1(q + out)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm_2(out2 + out3)
        return out3


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, d_model, device):
        super(Encoder, self).__init__()

        self.pos_emb = PositionalEncoding(d_model=d_model, device=device)
        self.num_layers = num_layers

        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, q, k, v):

        q = self.pos_emb(q)
        for i in range(self.num_layers):
            q = self.encoder_layers[i](q, k, v)

        return q


class Transformer(nn.Module):

    def __init__(self, input_size, d_model, nheads, num_layers, attn_type, seed, device="cpu"):

        super(Transformer, self).__init__()

        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)
        self.encoder_layer = EncoderLayer(d_model=d_model, attn_type=attn_type,
                                          n_heads=nheads, seed=seed)
        self.decoder_layer = DecoderLayer(d_model=d_model, attn_type=attn_type,
                                          n_heads=nheads, seed=seed)

        self.encoder = Encoder(self.encoder_layer, num_layers=num_layers, d_model=d_model, device=device)
        self.decoder = Decoder(self.decoder_layer, num_layers=num_layers, d_model=d_model, device=device)

    def forward(self, q_enc, k_enc, q_dec, k_dec):

        x_en = self.enc_embedding(q_enc)
        x_de = self.enc_embedding(q_dec)
        memory = self.encoder(x_en, k_enc, k_enc)
        output = self.decoder(x_de, k_dec, k_dec, memory)

        return memory, output


if __name__ == "__main__":

    transformer = Transformer(input_size=8, d_model=8, nheads=1, num_layers=1, attn_type="ATA", seed=1234)

    x_en = torch.randn(16, 24, 8)
    x_de = torch.randn(16, 24, 8)

    _, output = transformer(x_en, x_en, x_de)
    print(output)