import argparse
import os
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState
from torch import nn
from torch.optim import Adam

from GMM import GmmDiagonal
from Kmeans import Kmeans
from data_loader_userid import UserDataLoader
from deepclustering import DeepClustering
from mnist_data import MnistDataLoader
from psycology_data_loader import PatientDataLoader
from seed_manager import set_seed
from som_vae import SOMVAE
from synthetic_data import SyntheticDataLoader


set_seed(9800)


class Autoencoder(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, dim_rec=2):
        super(Autoencoder, self).__init__()
        self.n_layers = n_layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_rec)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_rec, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.encoder(x)
            x = self.decoder(x)
        return x


class DimRec:
    def __init__(self):
        super(DimRec, self).__init__()

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="synthetic")
        parser.add_argument("--model_name", type=str, default="ACAT")
        parser.add_argument("--num_epochs", type=int, default=10)
        parser.add_argument("--n_trials", type=int, default=10)
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--attn_type", type=str, default='basic')
        parser.add_argument("--max_encoder_length", type=int, default=96)
        parser.add_argument("--pred_len", type=int, default=24)
        parser.add_argument("--max_train_sample", type=int, default=-1)
        parser.add_argument("--max_test_sample", type=int, default=-1)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--var", type=int, default=1)
        parser.add_argument("--add_entropy", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--data_path", type=str, default='watershed.csv')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='no',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()
        self.seed = args.seed
        set_seed(self.seed)
        self.exp_name = args.exp_name
        self.var = args.var
        self.add_entropy = args.add_entropy

        if self.exp_name == "mnist":
            pass
        elif self.exp_name == "synthetic":
            pass
        elif self.exp_name == "User_id":

            data_path = "{}.csv".format(args.exp_name)
            data = pd.read_csv(data_path)
            data.sort_values(by=["id", "time"], inplace=True)

        else:
            data_path = "{}.csv".format(args.exp_name)
            data = pd.read_csv(data_path)

        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        print("using {}".format(self.device))
        self.exp_name = args.exp_name
        self.attn_type = args.attn_type
        self.num_iteration = args.max_train_sample
        self.max_encoder_length = args.max_encoder_length

        model_dir = "clustering_models_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.pred_len = args.pred_len
        self.model_name = "{}_{}_{}".format(args.model_name, args.exp_name, self.pred_len)
        self.model_path = model_dir
        self.cluster = args.cluster
        self.best_centroids = None

        if self.exp_name == "mnist":

            self.data_loader = MnistDataLoader(batch_size=args.batch_size, seed=self.seed)

        elif self.exp_name == "synthetic":

            self.data_loader = SyntheticDataLoader(batch_size=args.batch_size,
                                                   max_samples=args.max_train_sample,
                                                   seed=self.seed)

        elif self.exp_name == "User_id":
            self.data_loader = UserDataLoader(real_inputs=["time", "x", "y", "z"],
                                              max_encoder_length=args.max_encoder_length,
                                              max_train_sample=args.max_train_sample,
                                              batch_size=args.batch_size,
                                              device=self.device,
                                              data=data,
                                              seed=self.seed)
        else:

            self.data_loader = PatientDataLoader(max_encoder_length=args.max_encoder_length,
                                                 max_train_sample=args.max_train_sample,
                                                 batch_size=args.batch_size,
                                                 device=self.device,
                                                 data=data,
                                                 seed=self.seed)

        self.best_overall_valid_loss = 1e10
        self.best_dim_rec_model = nn.Module()

        self.n_clusters = self.data_loader.n_clusters
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.best_overall_valid_loss = 1e10
        self.list_explored_params = []

        self.red_path = "red_models"
        if not os.path.exists(self.red_path):
            os.makedirs(self.red_path)

        if args.model_name == "kmeans":
            Kmeans(n_clusters=self.n_clusters, batch_size=self.batch_size,
                   data_loader=self.data_loader.hold_out_test, seed=self.seed)
        else:
            self.best_dim_rec_model = nn.Module()
            self.run_optuna(args)
            self.dim_rec_rep()

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.model_name,
                                    direction="maximize")
        study.optimize(self.objective, n_trials=args.n_trials, n_jobs=4)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def train(self, trial):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
        tmax = trial.suggest_categorical("tmax", [100])
        n_layers = trial.suggest_categorical("n_layers", [1, 3])
        d_model_rec = trial.suggest_categorical("d_model_rec", [32, 64])

        tup_params = [tmax, d_model_rec, n_layers]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        dim_rec_model = Autoencoder(input_dim=self.data_loader.input_size * 96, hidden_dim=d_model_rec, n_layers=n_layers).to(self.device)

        d_model_list = [16, 32, 64, 128, 512]
        num_layers_list = [1, 3]
        knn_list = [20, 10, 5]
        gamma = [0.1, 0.01]

        best_trial_valid_loss = 1e10

        print("working on combos...")

        for knn in knn_list:
            for d_model in d_model_list:
                for num_layers in num_layers_list:
                    for gm in gamma:
                        try:
                            if "som_vae" in self.model_name:
                                clustering_model = SOMVAE(d_input=self.max_encoder_length,
                                                           d_channel=self.data_loader.input_size,
                                                           n_clusters=self.n_clusters,
                                                           d_latent=d_model,
                                                           device=self.device).to(self.device)
                            elif "gmm" in self.model_name:
                                clustering_model = GmmDiagonal(num_feat=self.data_loader.input_size,
                                                                num_dims=d_model,
                                                                num_components=self.n_clusters,
                                                                device=self.device).to(self.device)
                            else:
                                clustering_model = DeepClustering(input_size=self.data_loader.input_size,
                                                                   n_clusters=self.n_clusters,
                                                                   d_model=d_model,
                                                                   nheads=8,
                                                                   num_layers=num_layers,
                                                                   attn_type=self.attn_type,
                                                                   seed=self.seed,
                                                                   device=self.device,
                                                                   pred_len=self.pred_len,
                                                                   batch_size=self.batch_size,
                                                                   var=self.var,
                                                                   knns=knn,
                                                                   gamma=gm,
                                                                   add_entropy=self.add_entropy).to(self.device)

                            checkpoint = torch.load(
                                os.path.join(self.model_path, "{}_forecast.pth".format(self.model_name)),
                                map_location=self.device)

                            clustering_model.load_state_dict(checkpoint)

                            clustering_model.eval()

                            optimizer = Adam(dim_rec_model.parameters())
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

                            for epoch in range(self.num_epochs):

                                tot_train_loss = 0
                                dim_rec_model.train()
                                for x, y in self.data_loader.train_loader:
                                    with torch.no_grad():
                                        _, _, _, _, _, x_rec_cluster = clustering_model(x.to(self.device), y.to(self.device))

                                    optimizer.zero_grad()
                                    x_rec_cluster = x_rec_cluster.reshape(self.batch_size, -1)
                                    x_dim_rec = dim_rec_model(x_rec_cluster)
                                    loss = nn.MSELoss()(x_rec_cluster, x_dim_rec)
                                    loss.backward()
                                    tot_train_loss += loss.item()
                                    optimizer.step()
                                    scheduler.step()

                                tot_test_loss = 0
                                dim_rec_model.eval()
                                for x, y in self.data_loader.test_loader:
                                    with torch.no_grad():
                                        _, _, _, _, _, x_rec_cluster = clustering_model(x.to(self.device), y.to(self.device))

                                    x_rec_cluster = x_rec_cluster.reshape(self.batch_size, -1)
                                    x_dim_rec = dim_rec_model(x_rec_cluster)
                                    loss = nn.MSELoss()(x_rec_cluster, x_dim_rec)
                                    tot_test_loss += loss.item()
                                    if loss < best_trial_valid_loss:
                                        best_trial_valid_loss = loss
                                        if best_trial_valid_loss < self.best_overall_valid_loss:
                                            self.best_overall_valid_loss = best_trial_valid_loss
                                            self.best_dim_rec_model = dim_rec_model

                                            torch.save(self.best_dim_rec_model.state_dict(),
                                                       os.path.join(self.red_path,
                                                                    "{}_red.pth".format(self.model_name)))

                                if epoch % 5 == 0:
                                    print(f"epoch {epoch}, train_loss: {tot_train_loss :.3f}")
                                    print(f"epoch {epoch}, valid_loss: {tot_test_loss :.3f}")
                        except RuntimeError as e:
                            pass

        return best_trial_valid_loss

    def objective(self, trial):

        return self.train(trial)

    def dim_rec_rep(self):

        d_model_list = [16, 32, 64, 128, 512]
        num_layers_list = [1, 3]
        knn_list = [20, 10, 5]
        gamma = [0.1, 0.01]

        list_2d = []
        list_y = []

        for knn in knn_list:
            for d_model in d_model_list:
                for num_layers in num_layers_list:
                    for gm in gamma:
                        try:
                            if "som_vae" in self.model_name:
                                clustering_model = SOMVAE(d_input=self.max_encoder_length,
                                                          d_channel=self.data_loader.input_size,
                                                          n_clusters=self.n_clusters,
                                                          d_latent=d_model,
                                                          device=self.device).to(self.device)
                            elif "gmm" in self.model_name:
                                clustering_model = GmmDiagonal(num_feat=self.data_loader.input_size,
                                                               num_dims=d_model,
                                                               num_components=self.n_clusters,
                                                               device=self.device).to(self.device)
                            else:
                                clustering_model = DeepClustering(input_size=self.data_loader.input_size,
                                                                  n_clusters=self.n_clusters,
                                                                  d_model=d_model,
                                                                  nheads=8,
                                                                  num_layers=num_layers,
                                                                  attn_type=self.attn_type,
                                                                  seed=self.seed,
                                                                  device=self.device,
                                                                  pred_len=self.pred_len,
                                                                  batch_size=self.batch_size,
                                                                  var=self.var,
                                                                  knns=knn,
                                                                  gamma=gm,
                                                                  add_entropy=self.add_entropy).to(self.device)

                            checkpoint = torch.load(
                                os.path.join(self.model_path, "{}_forecast.pth".format(self.model_name)),
                                map_location=self.device)

                            clustering_model.load_state_dict(checkpoint)

                            clustering_model.eval()
                            print("Successful...")

                            for x, y in self.data_loader.hold_out_test:

                                with torch.no_grad():
                                    _, _, _, _, _, x_rec_cluster = clustering_model(x.to(self.device),
                                                                                    y.to(self.device))
                                    x_rec_cluster = x_rec_cluster.reshape(self.batch_size, -1)
                                    x_dim_rec = self.best_dim_rec_model.encoder(x_rec_cluster)
                                    list_2d.append(x_dim_rec)
                                    list_y.append(y[:, 0, 0])

                        except RuntimeError:
                            pass

        x_reconstructs = torch.cat(list_2d)
        label = torch.cat(list_y).to(torch.int)

        x_reconstructs = x_reconstructs.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        colors = plt.cm.tab20.colors

        if not os.path.exists("two_d_plots"):
            os.makedirs("two_d_plots")

        for i in range(len(label)):

            plt.scatter(x_reconstructs[i][0], x_reconstructs[i][1], color=colors[label[i]])

        plt.legend(labels=[f"class {i+1}" for i in range(self.n_clusters)])
        plt.tight_layout()
        plt.savefig("two_d_plots/{}.pdf".format(self.model_name))

    # knns = np.vstack(knns)
    # x_reconstructs = np.vstack(x_reconstructs)
    # test_x = torch.linspace(0, 1, 100)
    #
    # print("adj rand index %.3f" % statistics.mean(tot_adj_loss))
    #
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    # alpha_arr = 0.1 + 0.9 * (1 - torch.arange(x_reconstructs.shape[1]) / x_reconstructs.shape[1])
    #
    # path_to_pdfs = "populations"
    # if not os.path.exists(path_to_pdfs):
    #     os.makedirs(path_to_pdfs)
    #
    # def get_color(ind):
    #     r, g, b, _ = to_rgba(colors[ind])
    #     color = [(r, g, b, alpha) for alpha in alpha_arr]
    #     return color
    #
    # # Plot the clusters
    #
    # inds = np.random.randint(0, len(x_reconstructs), 32)
    #
    # for i in inds:
    #
    #     ids = knns[i]
    #     x_1 = x_reconstructs[i].squeeze()
    #
    #     plt.scatter(test_x, x_1, color=get_color(0))
    #
    #     x_os = [x_reconstructs[j] for j in ids]
    #     for k, x in enumerate(x_os):
    #
    #         plt.scatter(test_x, x, color=get_color(k+1))
    #
    #     # Set plot labels and legend
    #     plt.title('')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #
    #     patches = [plt.Line2D([0], [0], color=to_rgba(colors[j]), marker='o', markersize=5, linestyle='None') for j in range(len(ids))]
    #     labels = [f"Sample {j+1}" for j in range(len(ids))]
    #     plt.legend(handles=patches, labels=labels)
    #     plt.tight_layout()
    #     plt.savefig("{}/synthetic_{}_{}.pdf".format(path_to_pdfs, i, self.exp_name))
    #     plt.clf()
DimRec()
