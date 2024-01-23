import torch
import torch.nn as nn


class TrainableKMeans(nn.Module):
    def __init__(self, input_size, num_dim, num_clusters, pred_len):
        super(TrainableKMeans, self).__init__()

        self.embed = nn.Linear(input_size, num_dim)
        self.centroids = nn.Parameter(torch.randn(num_clusters, num_dim))
        self.embed2 = nn.Linear(num_clusters, 1)
        self.pred_len = pred_len

    def forward(self, x, y=None):

        loss = 0.0

        x = self.embed(x)
        # Calculate distances to centroids
        distances = torch.cdist(x, self.centroids)

        # Assign clusters based on minimum distances
        output = torch.softmax(-distances, dim=-1)

        output = self.embed2(output)

        if y is not None:

            loss = nn.MSELoss()(y, output[:, -self.pred_len:, :])

        return output, loss

