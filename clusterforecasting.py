import numpy as np
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.distributions as dist
from sklearn.decomposition import PCA
from GMM import GmmFull, GmmDiagonal
from modules.transformer import Transformer
from sklearn.cluster import KMeans
from torchmetrics.clustering import AdjustedRandScore
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


def assign_clusters(points, centroids, rate, device):
    """
    Assign each point to the nearest cluster centroid.

    Args:
        points (torch.Tensor): Tensor of shape (num_points, dimension) representing data points.
        centroids (torch.Tensor): Tensor of shape (num_clusters, dimension) representing cluster centroids.

    Returns:
        cluster_indices (torch.Tensor): Tensor of shape (num_points,) containing the index of the nearest centroid for each point.
    """
    # Compute squared distances between each point and each centroid
    num_clusters = centroids.shape[0]

    distances = torch.cdist(points, centroids, p=2)**2  # Shape: (num_points, num_clusters)
    # Assign each point to the cluster with the smallest distance
    # distances_norm = torch.randn_like(distances)
    # distances_log_prob = torch.log(torch.softmax(distances, dim=-1))
    # distances_norm_log_prob = torch.log(torch.softmax(distances_norm, dim=-1))
    # kl_loss = nn.functional.kl_div(distances_log_prob, distances_norm_log_prob, reduction="batchmean", log_target=True)
    kl_loss = torch.tensor(0, device=device)
    cluster_indices = torch.argmin(distances, dim=1)

    return cluster_indices, kl_loss


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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


class ClusterForecasting(nn.Module):

    def __init__(self, input_size, output_size, len_snippets,
                 d_model, nheads,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size, n_clusters):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.enc_embedding = nn.Linear(input_size, d_model)

        self.seq_model = Transformer(input_size=d_model, d_model=d_model,
                                     nheads=nheads, num_layers=num_layers,
                                     attn_type=attn_type, seed=seed, device=device)

        self.cluster_centers = nn.Parameter(torch.randn((n_clusters, input_size), device=device))
        self.auto_encoder = Autoencoder(input_dim=d_model, encoding_dim=d_model)

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_clusters = n_clusters

    def forward(self, x, y):

        x_enc = self.enc_embedding(x)
        seq_len = x.shape[1]
        # auto-regressive generative
        output = self.seq_model(x_enc)

        input_to_cluster = self.auto_encoder(output)

        diff = input_to_cluster.unsqueeze(1) - input_to_cluster.unsqueeze(0)
        diff = diff.permute(2, 0, 1, 3)

        dist = torch.einsum('lbcd,lbcd-> lbc', diff, diff)
        dist_softmax = torch.softmax(-dist, dim=-1)
        _, k_nearest = torch.topk(dist_softmax, k=self.num_clusters, dim=-1)

        total_indices = torch.arange(self.num_clusters, device=self.device).unsqueeze(0).\
            repeat(self.batch_size, 1).unsqueeze(0).repeat(seq_len, 1, 1)
        mask_cluster = k_nearest == total_indices
        mask_cluster = mask_cluster.permute(2, 0, 1)
        tot_sum = 0

        for i in range(self.num_clusters):
            tot_sum += dist[mask_cluster[i]].sum()

        y_c = y.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).squeeze(-1)
        y_c = y_c.permute(2, 0, 1)

        labels = y_c[torch.arange(seq_len)[:, None, None]
                     ,torch.arange(self.batch_size)[None, :, None], k_nearest]

        assigned_labels = torch.mode(labels, dim=-1).values
        assigned_labels = assigned_labels.reshape(-1)
        y = y.reshape(-1)

        adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))

        return tot_sum, adj_rand_index, [assigned_labels, input_to_cluster]
