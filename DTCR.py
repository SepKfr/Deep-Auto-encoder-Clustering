import numpy as np
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
from sklearn import metrics
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore
from torchmetrics import F1Score
from seed_manager import set_seed


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def kmeans_regularization_loss(H, k):
      """
      Args:
        H: torch.Tensor representing the learned representation matrix.
        F: torch.Tensor representing the cluster indicator matrix.
        J: The original loss function (e.g., torch.nn.MSELoss()).
      Returns:
        A torch.Tensor representing the combined loss.
      """
      U, _, _ = torch.svd(H)  # Perform SVD on H
      F = U[:, :k]  # Get the first k singular vectors as F (closed-form solution)

      H_t_H = torch.einsum('bd, nd-> bn', H, H)
      trace_term = torch.trace(H_t_H)
      F_t_H = torch.einsum('bn, bc -> bc', H_t_H, F)
      F_t_HF = torch.einsum('bn, bc -> nc', F_t_H, F)
      reg_term = torch.trace(F_t_HF)

      return trace_term + reg_term


class DTCR(nn.Module):

    def __init__(self, input_size,
                 d_model, n_clusters,
                 num_layers, seed,
                 device, batch_size):

        super(DTCR, self).__init__()

        set_seed(seed)

        self.device = device

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=d_model,
                               bidirectional=True, num_layers=num_layers)
        self.decoder = nn.LSTM(input_size=d_model*2, hidden_size=input_size,
                               bidirectional=False, num_layers=1)

        self.proj_down = nn.Sequential(nn.Linear(d_model*2, 128),
                                       nn.Linear(128, 2))
        self.batch_size = batch_size
        self.d_model = d_model
        self.input_size = input_size
        self.n_clusters = n_clusters

    def forward(self, x, y=None):

        s_l = x.shape[1]

        if len(x.shape) > 3:
            x = x.reshape(self.batch_size, s_l, -1)

        a = 0.2  # Hyperparameter for the number of time steps to shuffle
        num_shuffle = int(a * s_l)
        shuffle_mask = torch.randperm(s_l)[:num_shuffle]
        x_to_shuffle = x[:, shuffle_mask, :]
        shuffle_mask_2 = torch.randperm(num_shuffle)
        x_to_shuffle = x_to_shuffle[:, shuffle_mask_2, :]
        inds_no_shuffle = [idx for idx in range(s_l) if idx not in shuffle_mask]
        x_not_shuffle = x[:, inds_no_shuffle, :]

        fake_x = torch.cat([x_not_shuffle, x_to_shuffle], dim=1)
        combined_x = torch.cat([x, fake_x], dim=0)

        x_enc, _ = self.encoder(combined_x)

        y_hat_class = self.proj_down(x_enc)
        y_class = torch.cat([torch.zeros(self.batch_size, 2), torch.ones(self.batch_size, 2)], dim=0).to(self.device)
        y_class = y_class.to(torch.long)

        class_loss = nn.CrossEntropyLoss()(y_hat_class, y_class)

        x_rec, _ = self.decoder(x_enc)

        x_enc_kmeans = x_enc.reshape(self.batch_size, -1)
        kmeans_loss = kmeans_regularization_loss(x_enc_kmeans, self.n_clusters)

        rec_loss = nn.MSELoss()(x_rec, combined_x).mean()

        tot_loss = rec_loss + class_loss + kmeans_loss

        if y is not None:
            x_enc = x_enc.reshape(self.batch_size, -1)
            x_enc_kmeans_2 = x_enc.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(x_enc_kmeans_2)
            labels = kmeans.labels_
            assigned_labels = torch.from_numpy(labels).to(torch.long).to(self.device)
            y = y[:, 0, :].reshape(-1).to(torch.long)

            adj_rand_index = AdjustedRandScore()(assigned_labels, y)
            nmi = NormalizedMutualInfoScore()(assigned_labels, y)
            f1 = F1Score(task='multiclass', num_classes=self.n_clusters).to(self.device)(assigned_labels, y)
            p_score = purity_score(y.cpu().detach().numpy(), assigned_labels.cpu().detach().numpy())
        else:
            adj_rand_index = tot_loss
            nmi = tot_loss
            f1 = tot_loss
            p_score = tot_loss

        return tot_loss, adj_rand_index, nmi, f1, p_score, x_enc
