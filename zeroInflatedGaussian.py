import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import torch


class ZeroInflatedGaussian(nn.Module):

    def __init__(self, num_clusters,
                 batch_size,
                 num_segments,
                 seq_len,
                 d_model):
        super(ZeroInflatedGaussian, self).__init__()

        self.num_clusters = num_clusters
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_segments = num_segments
        self.batch_size = batch_size

        self.weights = nn.Parameter(torch.ones(num_clusters) / num_clusters, requires_grad=True)
        self.means = nn.Parameter(torch.randn(num_clusters, d_model), requires_grad=True)

        self.covariances = [nn.Parameter(torch.randn(d_model, d_model), requires_grad=True) for _ in range(self.num_clusters)]

        self.p = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):

        # Calculate the likelihood of each data point for the zero-inflated multivariate normal distribution
        log_likelihoods = []
        samples = []

        x_mean = x[:, :, :, 0]

        for i in range(self.num_clusters):

            covar = self.covariances[i]
            covar = torch.mm(covar, covar.t())
            covar.add_(torch.eye(self.d_model, device=covar.device))

            normal = MultivariateNormal(self.means[i], covar)

            sample = x * self.means[i] + covar.diagonal()
            samples.append(sample)

            # Point mass at zero
            point_mass_at_zero = (x_mean == 0).float() * (1 - self.p)

            # Gaussian distribution
            gaussian_prob = (1 - self.p) * normal.log_prob(x)

            # Combine components
            log_prob = torch.logsumexp(torch.stack([torch.log(1 - self.p) + torch.log(point_mass_at_zero),
                                                    gaussian_prob],
                                                   dim=0), dim=0)

            # log likelihood

            log_likelihoods.append(log_prob)

        log_likelihoods = torch.stack(log_likelihoods, dim=-1)
        samples = torch.stack(samples, dim=0)

        weighted_log_likelihoods = log_likelihoods + torch.log(self.weights)

        loss = -weighted_log_likelihoods.mean()

        return samples, loss