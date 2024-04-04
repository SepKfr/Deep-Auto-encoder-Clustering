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
from sklearn import metrics
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore
from torchmetrics import Accuracy
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


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


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

    def __init__(self, input_size, knns,
                 d_model, nheads, n_clusters,
                 num_layers, attn_type, seed,
                 device, pred_len, batch_size,
                 var=1):

        super(ClusterForecasting, self).__init__()

        self.device = device

        self.enc_embedding = nn.Linear(input_size, d_model)

        self.seq_model = Transformer(input_size=d_model, d_model=d_model,
                                     nheads=nheads, num_layers=num_layers,
                                     attn_type=attn_type, seed=seed, device=device)
        self.centers = nn.Parameter(torch.randn(n_clusters, d_model))
        self.proj_down = nn.Linear(d_model, input_size)

        self.pred_len = pred_len
        self.nheads = nheads
        self.batch_size = batch_size
        self.d_model = d_model
        self.input_size = input_size
        self.time_proj = 100
        self.num_clusters = n_clusters
        self.var = var
        self.k = knns

    def forward(self, x, y=None):

        x_enc = self.enc_embedding(x)
        # auto-regressive generative
        x_enc = self.seq_model(x_enc)
        s_l = x.shape[1]
        x_enc_re = x_enc.reshape(self.batch_size, -1)
        attn_score = torch.einsum('bl, cl-> bc', x_enc_re, x_enc_re) / np.sqrt(self.d_model * s_l)
        mask = torch.zeros_like(attn_score).fill_diagonal_(1).to(torch.bool)
        attn_score = attn_score.masked_fill(mask, value=-torch.inf)
        scores = torch.softmax(attn_score, dim=-1)

        x_rec = torch.einsum('bd, bc-> bd', x_enc_re, scores)
        x_rec = x_rec.reshape(x_enc.shape)
        x_rec_proj = self.proj_down(x_rec)

        _, top_scores = torch.topk(scores, k=self.k, dim=-1)
        x_rec_proj_exp = x_rec_proj.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        x_rec_proj_exp_se = x_rec_proj_exp[torch.arange(self.batch_size)[:, None],
                                           top_scores]

        diff_1 = (torch.diff(x_rec_proj_exp_se, dim=1)**2).mean()
        diff_2 = (torch.diff(x_rec_proj_exp_se, dim=2)**2).mean()

        loss = nn.MSELoss()(x_rec_proj, x) + diff_1 + diff_2 if self.var == 2 else nn.MSELoss()(x_rec_proj, x)

        #x_rec = self.proj_down(output_seq)

        # diffs = torch.diff(x_rec, dim=1)
        # kernel = 3
        # padding = (kernel - 1) // 2
        # mv_avg = nn.AvgPool1d(kernel_size=kernel, padding=padding, stride=1)(diffs.permute(0, 2, 1)).permute(0, 2, 1)
        # res = nn.MSELoss()(diffs, mv_avg)

        # dist_3d = torch.cdist(x_enc, self.centers, p=2)
        # _, cluster_ids = torch.min(dist_3d, dim=-1)
        #
        # assigned_centroids = self.centers[cluster_ids]
        #
        # distances = torch.norm(x_enc - assigned_centroids, dim=1)
        #
        # # Compute the mean squared distance
        # loss = torch.mean(distances ** 2)

        # mask = torch.zeros(self.batch_size, self.batch_size).to(self.device)
        # mask = mask.fill_diagonal_(1).to(torch.bool)
        # mask = mask.unsqueeze(-1).repeat(1, 1, s_l)
        # dist_3d = dist_3d.reshape(self.batch_size, self.batch_size, -1)
        #
        # dist_3d = dist_3d.masked_fill(mask, value=0.0)
        #
        # # x_rec = torch.einsum('lb, bd -> ld', dist_softmax, output_seq)
        # # x_rec_re = x_rec.reshape(self.batch_size, -1, self.d_model)
        # # x_rec_proj = self.proj_down(x_rec_re)
        # # loss_rec = nn.MSELoss()(x_rec_proj, x)
        #
        # knn_tensor, k_nearest = torch.topk(dist_3d, k=self.k, dim=1)


        # x_rec_expand = x_rec.unsqueeze(0).expand(self.batch_size, -1, -1)
        # k_nearest_e = k_nearest.unsqueeze(-1).repeat(1, 1, self.d_model)
        #
        # selected = x_rec_expand[torch.arange(self.batch_size)[:, None, None],
        #                         k_nearest_e,
        #                         torch.arange(self.d_model)[None, None, :]]
        #
        # selected = selected.reshape(self.batch_size, self.d_model, -1)
        #
        # diff_knns = (torch.diff(selected, dim=-1) ** 2).mean()
        # diff_steps = (torch.diff(selected, dim=1) ** 2).mean()

        #dist_knn = dist_softmax[torch.arange(self.batch_size)[:, None], k_nearest]

        #loss = loss_rec + diff_steps + diff_knns if self.var == 2 else loss_rec

        if y is not None:

            y = y[:, 0, :].reshape(-1)
            y_c = y.unsqueeze(0).expand(self.batch_size, -1)

            _, k_nearest = torch.topk(scores, k=self.k, dim=-1)
            labels = y_c[torch.arange(self.batch_size)[:, None],
                         k_nearest]

            assigned_labels = torch.mode(labels, dim=-1).values
            adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))
            nmi = NormalizedMutualInfoScore()(assigned_labels.to(torch.long), y.to(torch.long))
            acc = Accuracy(task='multiclass', num_classes=self.num_clusters).to(self.device)(assigned_labels.to(torch.long), y.to(torch.long))
            p_score = purity_score(y.to(torch.long).detach().cpu().numpy(), assigned_labels.to(torch.long).detach().cpu().numpy())

        else:

            adj_rand_index = torch.tensor(0, device=self.device)
            nmi = torch.tensor(0, device=self.device)
            acc = torch.tensor(0, device=self.device)
            p_score = torch.tensor(0, device=self.device)

        return loss, adj_rand_index, nmi, acc, p_score, x_rec_proj
