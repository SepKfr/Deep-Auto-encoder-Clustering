import argparse
import os
from itertools import product
import random
import statistics
import matplotlib.lines
from torchmetrics.clustering import AdjustedRandScore
import matplotlib.pyplot as plt
import optuna
from torch import nn
from torch.optim import Adam
import dataforemater
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from optuna.trial import TrialState
from torch.nn.utils import clip_grad_norm_
from GMM import GmmFull, GmmDiagonal
from clusterforecasting import ClusterForecasting
from forecasting import Forecasting
from data_loader import CustomDataLoader
from data_loader_userid import UserDataLoader
from Kmeans import TrainableKMeans
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba
from synthetic_data import SyntheticDataLoader


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Train:
    def __init__(self):

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="User_id")
        parser.add_argument("--model_name", type=str, default="basic_attn")
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--n_trials", type=int, default=10)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--attn_type", type=str, default='ATA')
        parser.add_argument("--max_encoder_length", type=int, default=24)
        parser.add_argument("--pred_len", type=int, default=24)
        parser.add_argument("--max_train_sample", type=int, default=32000)
        parser.add_argument("--max_test_sample", type=int, default=3840)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--var", type=int, default=1)
        parser.add_argument("--data_path", type=str, default='watershed.csv')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='no',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()
        self.exp_name = args.exp_name
        self.var = args.var

        if self.exp_name == "synthetic":
            pass
        elif self.exp_name == "User_id":
            data_path = "{}.csv".format(args.exp_name)
            data = pd.read_csv(data_path)
            data.sort_values(by=["id", "time"], inplace=True)
        else:
            self.data_formatter = dataforemater.DataFormatter(args.exp_name)
            # "{}.csv".format(args.exp_name)

            data_path = "{}.csv".format(args.exp_name)
            df = pd.read_csv(data_path, dtype={'date': str})
            df.sort_values(by=["id", "hours_from_start"], inplace=True)
            data = self.data_formatter.transform_data(df)

        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        print("using {}".format(self.device))
        self.exp_name = args.exp_name
        self.attn_type = args.attn_type
        self.num_iteration = args.max_train_sample
        self.max_encoder_length = args.max_encoder_length

        model_dir = "forecasting_models_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.pred_len = args.pred_len
        self.model_name = "{}_{}_{}".format(args.model_name, args.exp_name, self.pred_len)
        self.model_path = model_dir
        self.cluster = args.cluster
        self.best_centroids = None

        if self.exp_name == "synthetic":

            self.data_loader = SyntheticDataLoader(batch_size=args.batch_size, max_samples=args.max_train_sample)

        elif self.exp_name == "User_id":
            self.data_loader = UserDataLoader(real_inputs=["time", "x", "y", "z"],
                                              max_encoder_length=args.max_encoder_length,
                                              max_train_sample=args.max_train_sample,
                                              batch_size=args.batch_size,
                                              device=self.device,
                                              data=data,
                                              target_col="id")
        else:
            # Data loader configuration (replace with your own dataloader)
            self.data_loader = CustomDataLoader(real_inputs=self.data_formatter.real_inputs,
                                                max_encoder_length=args.max_encoder_length,
                                                pred_len=self.pred_len,
                                                max_train_sample=args.max_train_sample,
                                                max_test_sample=args.max_test_sample,
                                                batch_size=args.batch_size,
                                                device=self.device,
                                                data=data,
                                                target_col=self.data_formatter.target_column)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.best_overall_valid_loss = 1e-10
        self.list_explored_params = []

        self.best_forecasting_model = nn.Module()
        self.run_optuna(args)

        self.evaluate()

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

    def train_forecasting(self, trial):

        d_model = trial.suggest_categorical("d_model", [16, 32])
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        kernel = trial.suggest_categorical("kernel", [3])
        num_clusters = 4

        tup_params = [d_model, num_layers, kernel]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        if self.cluster == "yes":
            model = ClusterForecasting(input_size=self.data_loader.input_size,
                                       n_clusters=num_clusters,
                                       d_model=d_model,
                                       nheads=8,
                                       num_layers=num_layers,
                                       attn_type=self.attn_type,
                                       seed=1234,
                                       device=self.device,
                                       pred_len=self.pred_len,
                                       batch_size=self.batch_size,
                                       var=self.var).to(self.device)
        else:
            model = Forecasting(input_size=self.data_loader.input_size,
                                output_size=self.data_loader.output_size,
                                d_model=d_model,
                                nheads=8,
                                num_layers=num_layers,
                                attn_type=self.attn_type,
                                seed=1234,
                                device=self.device,
                                pred_len=self.pred_len,
                                batch_size=self.batch_size).to(self.device)

        forecast_optimizer = Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(forecast_optimizer, self.num_iteration)

        best_trial_valid_loss = 1e-10

        for epoch in range(self.num_epochs):

            list_of_valid_loss = []
            list_of_train_loss = []
            list_of_valid_adj = []
            list_of_train_adj = []

            for i in range(self.data_loader.n_folds):
                print(f"running on {i} fold...")

                model.train()
                train_knn_loss = 0
                train_adj_loss = 0

                for x, y in self.data_loader.list_of_train_loader[i]:

                    loss, adj_rand_index, _ = model(x.to(self.device), y.to(self.device))

                    forecast_optimizer.zero_grad()
                    loss.backward()
                    forecast_optimizer.step()
                    scheduler.step()
                    train_knn_loss += loss.item()
                    train_adj_loss += adj_rand_index.item()

                list_of_train_loss.append(train_knn_loss/self.data_loader.len_train)
                list_of_train_adj.append(train_adj_loss/self.data_loader.len_train)

                model.eval()
                valid_knn_loss = 0
                valid_adj_loss = 0

                for x, y in self.data_loader.list_of_test_loader[i]:

                    loss, adj_rand_index, _ = model(x.to(self.device), y.to(self.device))
                    valid_knn_loss += loss.item()
                    valid_adj_loss += adj_rand_index.item()

                list_of_valid_loss.append(valid_knn_loss/self.data_loader.len_test)
                list_of_valid_adj.append(valid_adj_loss/self.data_loader.len_test)

                if i == self.data_loader.n_folds - 1:
                    valid_tmp = statistics.mean(list_of_valid_adj)
                    if valid_tmp > best_trial_valid_loss:
                        best_trial_valid_loss = valid_tmp
                        if best_trial_valid_loss > self.best_overall_valid_loss:
                            self.best_overall_valid_loss = best_trial_valid_loss
                            self.best_forecasting_model = model
                            torch.save(self.best_forecasting_model.state_dict(),
                                       os.path.join(self.model_path,
                                                    "{}_forecast.pth".format(self.model_name)))

                if epoch % 5 == 0:
                    print("train KNN loss: {:.3f}, adj loss: {:.3f} epoch: {}"
                          .format(statistics.mean(list_of_train_loss),
                                  statistics.mean(list_of_train_adj), epoch))
                    print("valid KNN loss: {:.3f}, adj loss: {:.3f} epoch: {}"
                          .format(statistics.mean(list_of_valid_loss), statistics.mean(list_of_valid_adj), epoch))

        return best_trial_valid_loss

    def objective(self, trial):

        return self.train_forecasting(trial)

    def evaluate(self):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
        self.best_forecasting_model.eval()

        x_reconstructs = []
        inps = []
        knns = []
        tot_adj_loss = []

        for x, labels in self.data_loader.hold_out_test:
            _, adj_loss, outputs = self.best_forecasting_model(x.to(self.device), labels.to(self.device))
            x_reconstructs.append(outputs[1].detach().cpu().numpy())
            knns.append(outputs[0].detach().cpu().numpy())
            tot_adj_loss.append(adj_loss.item())

        knns = np.vstack(knns)
        x_reconstructs = np.vstack(x_reconstructs)
        test_x = torch.linspace(0, 1, 100)

        print("adj rand index %.3f" % statistics.mean(tot_adj_loss))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

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
Train()
