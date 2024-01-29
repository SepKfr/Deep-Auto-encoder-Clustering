import torch
import torch.nn as nn
import numpy as np


class TrainableKMeans(nn.Module):
    def __init__(self, input_size, num_dim, num_clusters, pred_len):
        super(TrainableKMeans, self).__init__()

        self.embed = nn.Linear(input_size, num_dim)

        self.centroids = nn.Parameter(torch.randn(num_clusters, num_dim))

        self.embed2 = nn.Linear(num_dim, 1)

        self.pred_len = pred_len

        self.num_dim = num_dim

    def forward(self, x, y=None):

        loss = 0.0

        x = self.embed(x)
        # Calculate distances to centroids
        distances = torch.cdist(x, self.centroids)

        # Assign clusters based on minimum distances

        dists = torch.softmax(-distances, dim=-1)

        output = torch.einsum('bsc, cd-> bsd', dists, self.centroids) / np.sqrt(self.num_dim)

        output2 = self.embed2(output)

        if y is not None:

            loss = nn.MSELoss()(y, output2[:, -self.pred_len:, :])

        return output2, loss

