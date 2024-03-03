import numpy as np
import random
import torch.nn as nn
import torch
from torch.nn import Linear
from GMM import GmmFull, GmmDiagonal
from modules.transformer import Transformer
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


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

        output = self.encoder(x)
        mean = output[:, :, :self.d_model]
        log_var = output[:, :, -self.d_model:]
        f_output = self.reparameterize(mean, log_var)
        return f_output, mean, log_var


class ClusteringLoss(torch.nn.Module):
    def __init__(self, margin=1.0, inter_weight=1.0, intra_weight=1.0):
        super(ClusteringLoss, self).__init__()
        self.margin = margin
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight

    @staticmethod
    def inter_cluster_loss(centers, margin):
        """Calculate inter-cluster loss"""
        # Calculate pairwise distances between cluster centers
        distances = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=2)
        # Exclude self-distances
        inter_loss = (torch.sum(torch.triu(distances, diagonal=1) < margin).float()) / (
                centers.size(0) * (centers.size(0) - 1) / 2)
        return inter_loss

    @staticmethod
    def intra_cluster_loss(features, centers):
        """Calculate intra-cluster loss"""
        # Compute distances between each feature vector and each cluster center
        distances = torch.norm(features.unsqueeze(1) - centers.unsqueeze(0), dim=2)
        # Assign each point to the nearest cluster
        _, assigned_clusters = torch.min(distances, dim=1)
        intra_losses = []
        # Calculate variance within each cluster
        for i in range(centers.size(0)):
            cluster_points = features[assigned_clusters == i]
            if len(cluster_points) > 1:
                intra_loss = torch.mean(torch.norm(cluster_points - torch.mean(cluster_points, dim=0), dim=1))
                intra_losses.append(intra_loss)
        # Take the average of intra-cluster losses
        if len(intra_losses) > 0:
            intra_loss = torch.mean(torch.stack(intra_losses))
        else:
            intra_loss = torch.tensor(0.0)
        return intra_loss

    def forward(self, features, centers):
        # Compute inter-cluster loss
        inter_loss = self.inter_weight * self.inter_cluster_loss(centers, self.margin)
        # Compute intra-cluster loss
        intra_loss = self.intra_weight * self.intra_cluster_loss(features, centers)
        # Total loss is the sum of both
        total_loss = inter_loss + intra_loss
        return total_loss


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size, len_snippets,
                 d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size, num_clusters=5):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.enc_embedding = nn.Linear(input_size, d_model)

        self.seq_model = Transformer(input_size=d_model, d_model=d_model,
                                     nheads=nheads, num_layers=num_layers,
                                     attn_type=attn_type, seed=seed, device=device)

        self.cluster_centers = nn.Parameter(torch.randn((num_clusters, d_model*len_snippets), device=device))

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model

    def forward(self, x, x_seg, y=None):

        x = self.enc_embedding(x_seg)
        output = self.seq_model(x)
        output = output.reshape(x.shape)
        input_to_cluster = output.reshape(self.batch_size * x.shape[1], -1)
        loss = ClusteringLoss()(input_to_cluster, self.cluster_centers)

        return loss

        # enc_output, mean, log_var = self.encoder(x)
        #
        # dec_output_cp = self.decoder(enc_output)
        #
        # final_out = self.fc_dec(enc_output)[:, -self.pred_len:, :]
        #
        # if y is not None:
        #
        #     output_cp = dec_output_cp.permute(0, 2, 1)
        #
        #     loss_cp = nn.CrossEntropyLoss()(output_cp, x_silver_standard)
        #
        #     kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        #
        #     cp_loss_total = loss_cp + torch.clip(self.w[0], min=0, max=0.01) * kl_divergence
        #
        #     mse_loss = nn.MSELoss()(y, final_out)
        #
        #     if self.training:
        #         loss_total = mse_loss + torch.clip(self.w[1], min=0, max=0.01) * cp_loss_total
        #     else:
        #         loss_total = mse_loss
        #
        # return final_out, loss_total
