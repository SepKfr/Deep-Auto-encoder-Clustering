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
                 device, pred_len, batch_size):

        super(ClusterForecasting, self).__init__()

        self.device = device
        self.enc_embedding = nn.Linear(input_size, d_model)

        self.seq_model = Transformer(input_size=d_model, d_model=d_model,
                                     nheads=nheads, num_layers=num_layers,
                                     attn_type=attn_type, seed=seed, device=device)
        self.proj_down = nn.Linear(d_model, input_size)

        #self.cluster_centers = nn.Parameter(torch.randn((n_clusters, input_size), device=device))
        self.auto_encoder = Autoencoder(input_dim=input_size, encoding_dim=d_model)
        self.gp_model = DeepGPp(num_inducing=batch_size, num_hidden_dims=2)

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model
        self.input_size = input_size
        self.time_proj = 100
        self.num_clusters = 5

    def forward(self, x, y=None):

        x_enc = self.enc_embedding(x)
        # auto-regressive generative
        output = self.seq_model(x_enc)

        x_rec = self.proj_down(output)

        x_rec = self.gp_model.predict(x_rec)

        x_1 = torch.zeros(self.batch_size, self.batch_size, x.shape[1], 2, device=self.device)

        for i in range(self.batch_size):
            x_1[i, :, :, :] = x_rec[i, :, :].unsqueeze(0).repeat(self.batch_size, 1, 1)

        x_2 = x_rec.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        x_1 = x_1.reshape(-1, x.shape[1], 2)
        x_2 = x_2.reshape(-1, x.shape[1], 2)

        dtw = SoftDTWLossPyTorch(gamma=0.1)
        dtw_dist = dtw(x_1, x_2)
        dtw_dist = dtw_dist.reshape(-1, self.batch_size)

        dist_softmax = torch.softmax(-dtw_dist, dim=-1)
        _, k_nearest = torch.topk(dist_softmax, k=self.num_clusters, dim=-1)

        dist_knn = dtw_dist[torch.arange(self.batch_size)[:, None], k_nearest]
        loss = dist_knn.mean()

        # if y is not None:
        #     y = y[:, -1, :]
        #     y_c = y.unsqueeze(0).repeat(self.batch_size, 1, 1).squeeze(-1)
        #
        #     labels = y_c[torch.arange(self.batch_size)[:, None], k_nearest]
        #
        #     assigned_labels = torch.mode(labels, dim=-1).values
        #     assigned_labels = assigned_labels.reshape(-1)
        #     y = y.reshape(-1)
        #
        #     adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))
        # else:
        #     adj_rand_index = torch.tensor(0, device=self.device)

        return loss, [k_nearest, x_rec]
