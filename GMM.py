import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Categorical, Independent, Normal, MixtureSameFamily


class GMM(nn.Module):

    def __init__(self, num_clusters, input_size, d_model):
        super(GMM, self).__init__()

        self.num_clusters = num_clusters
        self.d_model = d_model

        self.embedding = nn.Linear(input_size, d_model)

        self.weights = nn.Parameter(torch.ones(num_clusters), requires_grad=True)
        self.means = nn.Parameter(torch.randn(num_clusters, d_model), requires_grad=True)
        self.covariances = nn.Parameter(torch.randn(num_clusters, d_model), requires_grad=True)

    def forward(self, x, y=None, x_gmm=False):

        x = self.embedding(x)
        probs = torch.nn.functional.relu(self.weights)
        probs = probs / probs.sum()
        mixture = Categorical(probs)

        covar = nn.Softplus()(self.covariances)

        components = Independent(Normal(self.means, covar), 1)
        mixture_model = MixtureSameFamily(mixture, components)
        sample = mixture_model.sample(x.shape[:-1])

        log_porb = mixture_model.log_prob(x)
        loss = - torch.logsumexp(log_porb, dim=-1).mean()

        return sample, loss


