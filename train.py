import argparse
import os
from itertools import product
import random

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
from Kmeans import TrainableKMeans
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Train:
    def __init__(self):

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="solar")
        parser.add_argument("--model_name", type=str, default="basic_attn")
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--n_trials", type=int, default=10)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--attn_type", type=str, default='ATA')
        parser.add_argument("--max_encoder_length", type=int, default=192)
        parser.add_argument("--pred_len", type=int, default=24)
        parser.add_argument("--max_train_sample", type=int, default=1024)
        parser.add_argument("--max_test_sample", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--data_path", type=str, default='~/research/Corruption-resilient-Forecasting-Models/solar.csv')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='no',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()

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

        # Data loader configuration (replace with your own dataloader)
        self.data_loader = CustomDataLoader(real_inputs=[],
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
        self.best_overall_valid_loss = 1e10
        self.list_explored_params = []

        self.best_forecasting_model = nn.Module()
        self.run_optuna(args)

        self.evaluate()

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.model_name,
                                    direction="minimize")
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

        tup_params = [d_model, num_layers]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        if self.cluster == "yes":
            model = ClusterForecasting(input_size=self.data_loader.input_size,
                                       output_size=self.data_loader.output_size,
                                       len_snippets=self.data_loader.len_snippets,
                                       d_model=d_model,
                                       nheads=8,
                                       num_layers=num_layers,
                                       attn_type=self.attn_type,
                                       seed=1234,
                                       device=self.device,
                                       pred_len=self.pred_len,
                                       batch_size=self.batch_size).to(self.device)
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

        best_trial_valid_loss = 1e10
        for epoch in range(self.num_epochs):

            model.train()
            train_mse_loss = 0
            train_kl_loss = 0

            for x, x_seg, y in self.data_loader.train_loader:

                loss, kl_loss, _ = model(x.to(self.device), x_seg.to(self.device), y.to(self.device))

                forecast_optimizer.zero_grad()
                loss.backward()
                forecast_optimizer.step()
                scheduler.step()
                train_mse_loss += loss.item()
                train_kl_loss += kl_loss.item()

            model.eval()
            valid_loss = 0
            valid_kl_loss = 0

            for x, x_seg, valid_y in self.data_loader.valid_loader:

                loss, kl_loss, _ = model(x.to(self.device), x_seg.to(self.device), valid_y.to(self.device))
                valid_loss += loss.item()
                valid_kl_loss += kl_loss.item()

                if valid_loss < best_trial_valid_loss:
                    best_trial_valid_loss = valid_loss
                    if best_trial_valid_loss < self.best_overall_valid_loss:
                        self.best_overall_valid_loss = best_trial_valid_loss
                        self.best_forecasting_model = model
                        torch.save(self.best_forecasting_model.state_dict(),
                                   os.path.join(self.model_path,
                                                "{}_forecast.pth".format(self.model_name)))

            if epoch % 5 == 0:
                print(
                    "train MSE loss: {:.3f}, train KL loss: {:.3f} epoch: {}".format(train_mse_loss, train_kl_loss, epoch))
                print("valid MSE loss: {:.3f}, valid KL loss: {:.3f}".format(valid_loss, valid_kl_loss))

        return best_trial_valid_loss

    def objective(self, trial):

        return self.train_forecasting(trial)

    def evaluate(self):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
        self.best_forecasting_model.eval()

        cluster_assignments = []
        inputs_to_cluster = []

        for x, x_seg, test_y in self.data_loader.test_loader:

            _, _, outputs = self.best_forecasting_model(x=x.to(self.device), x_seg=x_seg.to(self.device))
            cluster_assignments.append(outputs[0])
            inputs_to_cluster.append(outputs[1])

        cluster_assignments = torch.cat(cluster_assignments, dim=0).detach().cpu().numpy()
        inputs_to_cluster = torch.cat(inputs_to_cluster, dim=0).detach().cpu().numpy()

        colors = ['r', 'g', 'b', 'c', 'm']

        # Plot the clusters
        for i in range(3):
            cluster_points = cluster_assignments == i
            plt.scatter(inputs_to_cluster[cluster_points, 0], inputs_to_cluster[cluster_points, 1], color=colors[i],
                        label=f'Cluster {i}')

        # Set plot labels and legend
        plt.title('Clustering Assignments')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig("cluster_assignments_{}.pdf".format(self.exp_name))

Train()
