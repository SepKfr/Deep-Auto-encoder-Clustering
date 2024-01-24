import torch
import torch.nn as nn


class TrainableKMeans(nn.Module):
    def __init__(self, input_size, num_dim, num_clusters, pred_len):
        super(TrainableKMeans, self).__init__()

        self.embed = nn.Linear(input_size, num_dim)

        self.centroids = nn.Parameter(torch.randn(num_clusters, num_dim))

        self.embed2 = nn.Linear(num_clusters, num_dim)

        self.embed3 = nn.Linear(num_dim, 1)

        self.pred_len = pred_len

    def predict(self, x):

        x = self.embed(x)
        # Calculate distances to centroids
        distances = torch.cdist(x, self.centroids)

        # Assign clusters based on minimum distances
        output = torch.softmax(-distances, dim=-1)

        output2 = self.embed2(output)

        return output2

    def forward(self, x, y=None):

        loss = 0.0

        x = self.embed(x)
        # Calculate distances to centroids
        distances = torch.cdist(x, self.centroids)

        # Assign clusters based on minimum distances
        output1 = torch.softmax(-distances, dim=-1)

        output2 = self.embed2(output1)

        output3 = self.embed3(output2)

        if y is not None:

            loss = nn.MSELoss()(y, output3[:, -self.pred_len:, :])

        return output3, loss

