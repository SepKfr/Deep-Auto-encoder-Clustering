import numpy as np
import torch
from sklearn import metrics
from torchmetrics import F1Score
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def get_scores(y, assigned_labels, n_clusters, device):

    adj_rand_index = AdjustedRandScore()(assigned_labels.to(torch.long), y.to(torch.long))
    nmi = NormalizedMutualInfoScore()(assigned_labels.to(torch.long), y.to(torch.long))
    f1 = F1Score(task='multiclass', num_classes=n_clusters).to(device)(assigned_labels.to(torch.long),
                                                                                 y.to(torch.long))
    p_score = purity_score(y.to(torch.long).detach().cpu().numpy(),
                           assigned_labels.to(torch.long).detach().cpu().numpy())

    return adj_rand_index, nmi, f1, p_score

