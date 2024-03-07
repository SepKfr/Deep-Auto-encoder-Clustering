import numpy as np
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
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


def assign_clusters(points, centroids):
    """
    Assign each point to the nearest cluster centroid.

    Args:
        points (torch.Tensor): Tensor of shape (num_points, dimension) representing data points.
        centroids (torch.Tensor): Tensor of shape (num_clusters, dimension) representing cluster centroids.

    Returns:
        cluster_indices (torch.Tensor): Tensor of shape (num_points,) containing the index of the nearest centroid for each point.
    """
    # Compute squared distances between each point and each centroid
    distances = torch.cdist(points, centroids, p=2)**2  # Shape: (num_points, num_clusters)

    # Assign each point to the cluster with the smallest distance
    cluster_indices = torch.argmin(distances, dim=1)

    cluster_count = torch.bincount(cluster_indices, minlength=centroids.size(0) + 1)
    loss = cluster_count.to(torch.float).std()

    return cluster_indices, loss


def compute_inter_cluster_loss(points, centroids, cluster_indices):
    """
    Compute the inter-cluster loss based on squared Euclidean distance.

    Args:
        points (torch.Tensor): Tensor of shape (num_points, dimension) representing data points.
        centroids (torch.Tensor): Tensor of shape (num_clusters, dimension) representing cluster centroids.
        cluster_indices (torch.Tensor): Tensor of shape (num_points,) containing the index of the nearest centroid for each point.

    Returns:
        inter_cluster_loss (torch.Tensor): Inter-cluster loss.
    """
    # Gather centroids corresponding to assigned clusters
    assigned_centroids = centroids[cluster_indices]  # Shape: (num_points, dimension)

    # Compute squared Euclidean distance between points and centroids
    inter_cluster_loss = torch.mean(torch.sum((points - assigned_centroids)**2, dim=1))

    return inter_cluster_loss


def compute_intra_cluster_loss(points, centroids, cluster_indices):
    """
    Compute the intra-cluster loss based on squared Euclidean distance.

    Args:
        points (torch.Tensor): Tensor of shape (num_points, dimension) representing data points.
        centroids (torch.Tensor): Tensor of shape (num_clusters, dimension) representing cluster centroids.
        cluster_indices (torch.Tensor): Tensor of shape (num_points,) containing the index of the nearest centroid for each point.

    Returns:
        intra_cluster_loss (torch.Tensor): Intra-cluster loss.
    """
    intra_cluster_losses = []
    for i in range(centroids.size(0)):
        # Mask points belonging to cluster i
        mask = cluster_indices == i
        cluster_points = points[mask]

        # Compute squared Euclidean distance between cluster points and centroid
        if len(cluster_points) > 0:
            centroid = centroids[i].unsqueeze(0)  # Add singleton dimension to match cluster_points
            distances = torch.sum((cluster_points - centroid)**2, dim=1)
            intra_cluster_loss_i = torch.mean(distances)
            intra_cluster_losses.append(intra_cluster_loss_i)

    # Compute overall intra-cluster loss as the mean of individual cluster losses
    intra_cluster_loss = torch.mean(torch.stack(intra_cluster_losses))

    return intra_cluster_loss


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
        self.w_loss = nn.Parameter(torch.softmax((torch.randn(3, device=device)), dim=0))

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model

    def forward(self, x, x_seg, y=None):

        x = self.enc_embedding(x_seg)
        output = self.seq_model(x)
        output = output.reshape(x.shape)
        input_to_cluster = output.reshape(self.batch_size * x.shape[1], -1)

        cluster_indices, entropy_loss = assign_clusters(input_to_cluster, self.cluster_centers)

        # Compute inter-cluster loss
        inter_loss = compute_inter_cluster_loss(input_to_cluster, self.cluster_centers, cluster_indices)

        # Compute intra-cluster loss
        intra_loss = compute_intra_cluster_loss(input_to_cluster, self.cluster_centers, cluster_indices)

        loss = self.w_loss[0] * inter_loss + self.w_loss[1] * intra_loss + self.w_loss[2] * entropy_loss

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
