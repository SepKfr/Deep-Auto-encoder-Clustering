import statistics

from sklearn.cluster import KMeans
import torch
import numpy as np
import random
from sklearn import metrics
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore
from torchmetrics import Accuracy

np.random.seed(1234)
random.seed(1234)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class Kmeans:
    def __init__(self, n_clusters, batch_size, data_loader):

        self.n_clusters = n_clusters
        self.batch_size = batch_size

        tot_adj_loss = []
        tot_acc_loss = []
        tot_nmi_loss = []
        tot_p_loss = []

        for x, y in data_loader:
            adj, nmi, acc, p_score = self.predict(x, y)
            tot_adj_loss.append(adj.item())
            tot_acc_loss.append(acc.item())
            tot_nmi_loss.append(nmi.item())
            tot_p_loss.append(p_score.item())

        adj = statistics.mean(tot_adj_loss)
        nmi = statistics.mean(tot_nmi_loss)
        acc = statistics.mean(tot_acc_loss)
        p_score = statistics.mean(tot_p_loss)

        print("adj rand index {:.3f}, nmi {:.3f}, acc {:.3f}, p_score {:.3f}".format(adj, nmi, acc, p_score))

    def predict(self, x, y):

        x = x.reshape(self.batch_size, -1)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1234, n_init="auto").fit(x)
        labels = kmeans.labels_
        assigned_labels = torch.from_numpy(labels).to(torch.long)
        y = y[:, 0, :].reshape(-1).to(torch.long)

        adj_rand_index = AdjustedRandScore()(assigned_labels, y)
        nmi = NormalizedMutualInfoScore()(assigned_labels, y)
        acc = Accuracy(task='multiclass', num_classes=self.n_clusters)(assigned_labels, y)
        p_score = purity_score(y.detach().numpy(), assigned_labels.detach().numpy())

        return adj_rand_index, nmi, acc, p_score



