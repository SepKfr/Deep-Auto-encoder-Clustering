import statistics

from sklearn.cluster import KMeans
import torch
import numpy as np
import random
from sklearn import metrics
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore
from torchmetrics import F1Score
from seed_manager import set_seed


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class Kmeans:
    def __init__(self, n_clusters, batch_size, data_loader, seed, exp_name):

        set_seed(seed)
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
        f1 = statistics.mean(tot_acc_loss)
        p_score = statistics.mean(tot_p_loss)

        print("adj rand index {:.3f}, nmi {:.3f}, f1 {:.3f}, p_score {:.3f}".format(adj, nmi, f1, p_score))

        # Specify the file path
        file_path = "new_scores_{}_{}.csv".format(exp_name, self.seed)

        scores = {self.model_name: {'adj': f"{adj:.3f}",
                                    'f1': f"{f1: .3f}",
                                    'nmi': f"{nmi: .3f}",
                                    'p_score': f"{p_score: .3f}"}}

        df = pd.DataFrame.from_dict(scores, orient='index')

        if os.path.exists(file_path):

            df_old = pd.read_csv(file_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(file_path)
        else:
            df.to_csv(file_path)

    def predict(self, x, y):

        x = x.reshape(self.batch_size, -1)
        print(self.n_clusters)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1234, n_init="auto").fit(x)
        labels = kmeans.labels_
        assigned_labels = torch.from_numpy(labels).to(torch.long)
        y = y[:, 0, :].reshape(-1).to(torch.long)

        adj_rand_index = AdjustedRandScore()(assigned_labels, y)
        nmi = NormalizedMutualInfoScore()(assigned_labels, y)
        acc = F1Score(task='multiclass', num_classes=self.n_clusters)(assigned_labels, y)
        p_score = purity_score(y.detach().numpy(), assigned_labels.detach().numpy())

        return adj_rand_index, nmi, acc, p_score



