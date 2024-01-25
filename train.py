import argparse
import os
from itertools import product
import random
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
from data_loader import CustomDataLoader
from Kmeans import TrainableKMeans

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class NoamOpt:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Train:
    def __init__(self, pred_len):

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="solar")
        parser.add_argument("--model_name", type=str, default="basic_attn")
        parser.add_argument("--num_epochs", type=int, default=50)
        parser.add_argument("--n_trials", type=int, default=10)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--attn_type", type=str, default='autoformer')
        parser.add_argument("--max_encoder_length", type=int, default=192)
        parser.add_argument("--max_train_sample", type=int, default=32000)
        parser.add_argument("--max_test_sample", type=int, default=3840)
        parser.add_argument("--batch_size", type=int, default=3840)
        parser.add_argument("--data_path", type=str, default='~/research/Corruption-resilient-Forecasting-Models/solar.csv')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='yes',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()

        data_formatter = dataforemater.DataFormatter(args.exp_name)

        df = pd.read_csv("{}.csv".format(args.exp_name), dtype={'date': str})
        df.sort_values(by=["id", "hours_from_start"], inplace=True)
        data = data_formatter.transform_data(df)

        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        print("using {}".format(self.device))
        self.exp_name = args.exp_name
        self.attn_type = args.attn_type
        self.num_iteration = args.max_train_sample
        self.max_encoder_length = args.max_encoder_length

        model_dir = "forecasting_models_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_name = "{}+{}_{}".format(args.model_name, args.exp_name, pred_len)
        self.model_path = model_dir
        self.cluster = args.cluster
        self.pred_len = pred_len

        # Data loader configuration (replace with your own dataloader)
        self.data_loader = CustomDataLoader(real_inputs=[],
                                            max_encoder_length=args.max_encoder_length,
                                            pred_len=self.pred_len,
                                            max_train_sample=args.max_train_sample,
                                            max_test_sample=args.max_test_sample,
                                            batch_size=args.batch_size,
                                            device=self.device,
                                            data=data,
                                            target_col=data_formatter.target_column)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.best_overall_valid_loss = 1e10
        self.list_explored_params = []
        self.cluster_type = "kmeans"

        if self.cluster == "yes":

            self.best_forecasting_model = nn.Module()
            self.best_cluster_model = nn.Module()

            self.run_optuna(args)
            self.best_overall_valid_loss = 1e10
            self.list_explored_params = []
            self.cluster = "no"
            self.run_optuna(args)

        else:
            self.best_forecasting_model = nn.Module()
            self.best_cluster_model = None
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

    def train_kmeans(self, trial):

        d_model = trial.suggest_categorical("d_model", [16, 32])
        num_clusters = trial.suggest_categorical("num_clusters", [3, 5])
        tup_params = [d_model, num_clusters]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        model = TrainableKMeans(num_clusters=num_clusters,
                                input_size=self.data_loader.input_size,
                                num_dim=d_model,
                                pred_len=self.pred_len).to(self.device)

        optimizer = Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_iteration)

        best_trial_valid_loss = 1e10

        for epoch in range(self.num_epochs):

            model.train()
            train_mse = 0

            for x, y in self.data_loader.train_loader:

                output, loss = model(x.to(self.device), y.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_mse += loss.item()

            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            model.eval()
            valid_loss = 0

            for x, y in self.data_loader.valid_loader:

                output, loss = model(x.to(self.device), y.to(self.device))
                valid_loss += loss.item()

                if valid_loss < best_trial_valid_loss:
                    best_trial_valid_loss = valid_loss
                    if best_trial_valid_loss < self.best_overall_valid_loss:
                        self.best_overall_valid_loss = best_trial_valid_loss

                        self.best_cluster_model = model
                        torch.save(self.best_cluster_model.state_dict(),
                                   os.path.join(self.model_path, "{}_cluster.pth".format(self.model_name)))

            if epoch % 5 == 0:
                print("train MSE loss: {:.3f} epoch: {}".format(train_mse, epoch))
                print("valid loss: {:.3f}".format(valid_loss))

        return best_trial_valid_loss

    def train_gmm(self, trial):

        d_model = trial.suggest_categorical("d_model", [16, 32])
        num_clusters = trial.suggest_categorical("num_clusters", [3, 5])
        w_steps = trial.suggest_categorical("w_steps", [4000, 8000])
        tup_params = [d_model, num_clusters, w_steps]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        model = GmmFull(num_components=num_clusters, num_dims=d_model, num_feat=self.data_loader.input_size).to(self.device)
        component_optimizer = Adam(model.component_parameters())
        mixture_optimizer = Adam(model.mixture_parameters())

        mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, self.num_iteration)
        component_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(component_optimizer, self.num_iteration)

        clip_grad_norm_(model.component_parameters(), max_norm=1.0)
        clip_grad_norm_(model.mixture_parameters(), max_norm=1.0)

        best_trial_valid_loss = 1e10

        for epoch in range(self.num_epochs):

            model.train()
            train_nll = 0

            for x, y in self.data_loader.train_loader:

                loss, sample = model(x.to(self.device))

                component_optimizer.zero_grad()
                mixture_optimizer.zero_grad()
                loss.backward()
                mixture_optimizer.step()
                mixture_scheduler.step()
                component_optimizer.step()
                component_scheduler.step()
                model.constrain_parameters()
                train_nll += loss.item()

            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            model.eval()
            valid_loss = 0

            for x, valid_y in self.data_loader.valid_loader:

                loss, sample = model(x.to(self.device))
                valid_loss += loss.item()

                if valid_loss < best_trial_valid_loss:
                    best_trial_valid_loss = valid_loss
                    if best_trial_valid_loss < self.best_overall_valid_loss:
                        self.best_overall_valid_loss = best_trial_valid_loss
                        self.best_cluster_model = model
                        torch.save(self.best_cluster_model.state_dict(),
                                   os.path.join(self.model_path, "{}_cluster.pth".format(self.model_name)))

            if epoch % 5 == 0:
                print("train NLL loss: {:.3f} epoch: {}".format(train_nll, epoch))
                print("valid loss: {:.3f}".format(valid_loss))

        return best_trial_valid_loss

    def train_forecasting(self, trial):

        d_model = trial.suggest_categorical("d_model", [16, 32])
        num_layers = trial.suggest_categorical("num_layers", [1, 2] if
                                                self.cluster == "yes" else [1, 2, 3])
        w_steps = trial.suggest_categorical("w_steps", [4000, 8000])

        tup_params = [d_model, num_layers, w_steps]

        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        opt_num_dim = 0
        if self.best_cluster_model is not None:
            combination = list(product([16, 32], [3, 5]))

            for comb in combination:
                num_dim, num_cluster = comb
                try:
                    cluster_model = TrainableKMeans(num_dim=num_dim,
                                                    input_size=self.data_loader.input_size,
                                                    num_clusters=num_cluster,
                                                    pred_len=self.pred_len)
                    cluster_model.load_state_dict(torch.load(os.path.join(self.model_path,
                                                                         "{}_cluster.pth".format(self.model_name))))
                    cluster_model.to(self.device)
                    opt_num_dim = num_dim

                except RuntimeError:
                    pass
        else:
            cluster_model = None

        model = ClusterForecasting(input_size=self.data_loader.input_size,
                                   output_size=self.data_loader.output_size,
                                   d_model=d_model,
                                   nheads=8,
                                   num_layers=num_layers,
                                   attn_type=self.attn_type,
                                   seed=1234,
                                   device=self.device,
                                   pred_len=96,
                                   batch_size=self.batch_size,
                                   cluster_model=cluster_model,
                                   cluster_num_dim=opt_num_dim).to(self.device)

        forecast_optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

        best_trial_valid_loss = 1e10
        for epoch in range(self.num_epochs):

            model.train()
            train_mse_loss = 0

            for x, y in self.data_loader.train_loader:

                output, loss, _ = model(x.to(self.device), y.to(self.device))

                forecast_optimizer.zero_grad()
                loss.backward()
                forecast_optimizer.step_and_update_lr()
                train_mse_loss += loss.item()

            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            model.eval()
            valid_loss = 0

            for x, valid_y in self.data_loader.valid_loader:

                output, loss, _ = model(x.to(self.device), valid_y.to(self.device))
                valid_loss += loss.item()

                if valid_loss < best_trial_valid_loss:
                    best_trial_valid_loss = valid_loss
                    if best_trial_valid_loss < self.best_overall_valid_loss:
                        self.best_overall_valid_loss = best_trial_valid_loss
                        self.best_forecasting_model = model
                        torch.save(self.best_forecasting_model.state_dict(),
                                   os.path.join(self.model_path, "{}_forecast.pth".format(self.model_name)))

            if epoch % 5 == 0:
                print(
                    "train MSE loss: {:.3f} epoch: {}".format(train_mse_loss, epoch))
                print("valid loss: {:.3f}".format(valid_loss))

        return best_trial_valid_loss

    def objective(self, trial):

        if self.cluster == "yes":
            return self.train_kmeans(trial)
        else:
            return self.train_forecasting(trial)

    def evaluate(self):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
        self.best_forecasting_model.eval()
        mse_loss = 0
        mae_loss = 0

        for x, test_y in self.data_loader.test_loader:

            output, mse_l, mae_l = self.best_forecasting_model(x=x.to(self.device),
                                                               y=test_y.to(self.device))
            mse_loss += mse_l
            mae_loss += mae_l

        mse_loss = mse_loss / len(self.data_loader.test_loader)
        mae_loss = mae_loss / len(self.data_loader.test_loader)
        errors = {self.model_name: {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}"}}
        print(errors)

        error_path = "reported_errors_{}.csv".format(self.exp_name)

        df = pd.DataFrame.from_dict(errors, orient='index')

        if os.path.exists(error_path):
            df_old = pd.read_csv(error_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(error_path)
        else:
            df.to_csv(error_path)


for pred_len in [24, 48, 72, 96]:
    Train(pred_len=pred_len)
