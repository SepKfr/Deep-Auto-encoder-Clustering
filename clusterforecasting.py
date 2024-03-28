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
from tslearn.metrics import SoftDTWLossPyTorch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from GMM import GmmDiagonal

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32, mean_type='linear'):

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPp(DeepGP):
    def __init__(self, num_hidden_dims, num_inducing):

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=num_hidden_dims,
            mean_type='linear',
            num_inducing=num_inducing
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_hidden_dims)

    def forward(self, inputs):

        dist = self.hidden_layer(inputs)
        return dist

    def predict(self, x):

        dist = self(x)
        preds = self.likelihood(dist)
        preds_mean = preds.mean.mean(0)

        return preds_mean


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


class ClusterForecasting(nn.Module):

    def __init__(self, input_size,
                 d_model, nheads, n_clusters,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size, kernel):

        super(ClusterForecasting, self).__init__()

        self.device = device
        f = 9

        self.conv = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                  kernel_size=f, padding=int((f - 1) / 2)),
                                  nn.BatchNorm1d(d_model),
                                  nn.ReLU())

        self.enc_embedding = nn.Linear(input_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gmm = GmmDiagonal(num_components=n_clusters,
                               num_dims=d_model,
                               num_feat=d_model,
                               )

        self.seq_model = Transformer(input_size=d_model, d_model=d_model,
                                     nheads=nheads, num_layers=num_layers,
                                     attn_type=attn_type, seed=seed, device=device)
        self.proj_down = nn.Linear(d_model, 1)

        #self.cluster_centers = nn.Parameter(torch.randn((n_clusters, input_size), device=device))
        self.auto_encoder = Autoencoder(input_dim=input_size, encoding_dim=d_model)
        self.gp_model = DeepGPp(num_inducing=batch_size, num_hidden_dims=d_model)

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model
        self.input_size = input_size
        self.time_proj = 100
        self.num_clusters = 4

    def forward(self, x, y=None):

        x_enc = self.enc_embedding(x)
        # auto-regressive generative
        output_seq = self.seq_model(x_enc)
        gmm_loss = self.gmm(output_seq)
        cluster_assign = self.gmm.get_cluster_assign(output_seq)
        assigned_labels = torch.mode(cluster_assign, dim=-1).values

        y = y.reshape(-1)

        with torch.no_grad():
            adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))

        #x_rec = self.proj_down(output_seq)

        # x_rec = output_seq
        # diffs = torch.diff(x_rec, dim=1)
        # kernel = 3
        # padding = (kernel - 1) // 2
        # mv_avg = nn.AvgPool1d(kernel_size=kernel, padding=padding, stride=1)(diffs.permute(0, 2, 1)).permute(0, 2, 1)
        # res = nn.MSELoss()(diffs, mv_avg)
        #
        # x_dist = x_rec.reshape(self.batch_size, -1)
        #
        # diff = x_dist.unsqueeze(1) - x_dist.unsqueeze(0)
        #
        # dist_2d = torch.einsum('lbd,lbd-> lb', diff, diff)
        #
        # dist_softmax = torch.softmax(-dist_2d, dim=-1)
        # _, k_nearest = torch.topk(dist_softmax, k=self.num_clusters, dim=-1)
        #
        # x_rec_expand = x_dist.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # selected = x_rec_expand[torch.arange(self.batch_size)[:, None], k_nearest]
        #
        # diff_knns = (torch.diff(selected, dim=1) ** 2).sum()
        #
        # dist_knn = dist_2d[torch.arange(self.batch_size)[:, None], k_nearest]
        #
        # loss = dist_knn.sum() # + res.sum() + diff_knns
        # if y is not None:
        #
        #     y_c = y.unsqueeze(0).repeat(self.batch_size, 1, 1).squeeze(-1)
        #
        #     labels = y_c[torch.arange(self.batch_size)[:, None], k_nearest]
        #
        #     assigned_labels = torch.mode(labels, dim=-1).values
        #     assigned_labels = assigned_labels.reshape(-1)
        #     y = y.reshape(-1)
        #
        #     adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))
        #
        # else:
        #     adj_rand_index = torch.tensor(0, device=self.device)

        return gmm_loss, adj_rand_index
