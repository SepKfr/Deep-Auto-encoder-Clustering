from sklearn.cluster import KMeans
import torch
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
    def __init__(self, n_clusters, batch_size):

        self.n_clusters = n_clusters
        self.batch_size = batch_size

    def predict(self, x, y):

        x = x.reshape(self.batch_size, -1)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1234, n_init="auto").fit(x)
        labels = kmeans.labels_
        labels = torch.from_numpy(labels)
        assigned_labels = torch.mode(labels, dim=-1).values.to(torch.long)
        y = y[:, 0, :].reshape(-1).to(torch.long)

        adj_rand_index = AdjustedRandScore()(assigned_labels, y)
        nmi = NormalizedMutualInfoScore()(assigned_labels, y)
        acc = Accuracy(task='multiclass', num_classes=self.num_clusters)(assigned_labels, y)
        p_score = purity_score(y.to(torch.long).detach().numpy(), assigned_labels.detach().numpy())

        print("adj rand index {:.3f}, nmi {:.3f}, acc {:.3f}, p_score {:.3f}".format(adj_rand_index, nmi, acc, p_score))

