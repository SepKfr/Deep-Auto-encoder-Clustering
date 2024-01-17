import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical, Independent, Normal, MixtureSameFamily

import torch


class GMM(nn.Module):

    def __init__(self, num_clusters,
                 d_model):
        super(GMM, self).__init__()

        self.num_clusters = num_clusters
        self.d_model = d_model

        self.weights = nn.Parameter(torch.ones(num_clusters) / num_clusters, requires_grad=True)
        self.means = nn.Parameter(torch.randn(num_clusters, d_model), requires_grad=True)
        self.covariances = nn.Parameter(torch.randn(num_clusters, d_model, d_model), requires_grad=True)

    def forward(self, x):

        log_likelihoods = []

        for i in range(self.num_clusters):

            covar = self.covariances[i]
            covar = torch.mm(covar, covar.t())
            covar.add_(torch.eye(self.d_model, device=x.device))
            dist = torch.distributions.MultivariateNormal(self.means[i], covar)
            log_likelihoods.append(dist.log_prob(x))

        log_likelihood = torch.stack(log_likelihoods, dim=-1)

        weighted_log_prob = log_likelihood + torch.log(self.weights)

        p_k = torch.argmax(torch.exp(weighted_log_prob), dim=-1)

        one_hot = torch.nn.functional.one_hot(p_k, num_classes=self.num_clusters).float()
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.num_clusters)
        one_hot = one_hot.unsqueeze(2).repeat(1, 1, self.d_model, 1)
        x = (x * one_hot).mean(dim=1)
        x = x.permute(0, 2, 1)

        loss = - weighted_log_prob.mean()

        return x, loss


