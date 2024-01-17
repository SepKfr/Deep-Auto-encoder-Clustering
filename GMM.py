import torch.nn as nn
import torch
from torch.distributions import Categorical, Independent, Normal, MixtureSameFamily


class GMM(nn.Module):

    def __init__(self, num_clusters,
                 d_model):
        super(GMM, self).__init__()

        self.num_clusters = num_clusters
        self.d_model = d_model

        self.weights = nn.Parameter(torch.ones(num_clusters), requires_grad=True)
        self.means = nn.Parameter(torch.randn(num_clusters, d_model), requires_grad=True)
        self.covariances = nn.Parameter(torch.randn(num_clusters, d_model, d_model), requires_grad=True)

    def forward(self, x):

        probs = torch.nn.functional.relu(self.weights)
        probs = probs / probs.sum()
        mixture = Categorical(probs)

        covar_init = self.covariances

        covar_init_trans = torch.transpose(covar_init, 1, 2)
        covar = torch.einsum('cdb, cbd -> cd', covar_init, covar_init_trans)

        components = Independent(Normal(self.means, covar), 1)
        mixture_model = MixtureSameFamily(mixture, components)
        sample = mixture_model.sample(x.shape[:-1])

        loss = - mixture_model.log_prob(x).mean()

        x = x + sample

        return x, loss


