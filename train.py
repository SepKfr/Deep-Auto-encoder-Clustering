import argparse
import os
import optuna
from torch import nn
from torch.optim import Adam
import dataforemater
import pandas as pd
import torch
import torch.nn.functional as F
from optuna.trial import TrialState

from GMM import GMM
from clusterforecasting import ClusterForecasting
from data_loader import CustomDataLoader
from forecasting import Forecasting


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
    def __init__(self):

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="traffic")
        parser.add_argument("--model_name", type=str, default="basic_attn_forecast")
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--n_trials", type=int, default=10)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--attn_type", type=str, default='autoformer')
        parser.add_argument("--pred_len", type=int, default=96)
        parser.add_argument("--max_encoder_length", type=int, default=192)
        parser.add_argument("--max_train_sample", type=int, default=32000)
        parser.add_argument("--max_test_sample", type=int, default=3840)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--data_path", type=str, default='~/research/Corruption-resilient-Forecasting-Models/solar.csv')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='yes',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()

        data_formatter = dataforemater.DataFormatter(args.exp_name)

        df = pd.read_csv(args.data_path, dtype={'date': str})
        df.sort_values(by=["id", "hours_from_start"], inplace=True)
        data = data_formatter.transform_data(df)

        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        print("using {}".format(self.device))
        self.exp_name = args.exp_name
        self.attn_type = args.attn_type
        self.max_encoder_length = args.max_encoder_length

        model_dir = "forecasting_models_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_name = "{}_{}".format(args.model_name, args.pred_len)
        self.model_path = model_dir
        self.cluster = args.cluster
        self.pred_len = args.pred_len

        # Data loader configuration (replace with your own dataloader)
        self.data_loader = CustomDataLoader(real_inputs=[],
                                            max_encoder_length=args.max_encoder_length,
                                            pred_len=args.pred_len,
                                            max_train_sample=args.max_train_sample,
                                            max_test_sample=args.max_test_sample,
                                            batch_size=args.batch_size,
                                            device=self.device,
                                            data=data,
                                            target_col=data_formatter.target_column)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.best_overall_valid_loss = 1e10
        self.best_model = nn.Module()
        self.list_explored_params = []
        self.num_optuna_run = 1
        if self.cluster == "yes":
            self.GMM_best_model = nn.Module()
            self.run_optuna(args)
            self.num_optuna_run = 2
            self.cluster = "no"
            self.best_overall_valid_loss = 1e10
            self.run_optuna(args)
        else:
            self.GMM_best_model = None
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

    def objective(self, trial):

        d_model = trial.suggest_categorical("d_model", [16, 32])
        num_layers = trial.suggest_categorical("num_layers", [1, 2] if self.cluster == "no" else [1])
        num_clusters = trial.suggest_categorical("num_clusters", [3, 5] if
                                                 self.cluster == "yes" else [1])
        w_steps = trial.suggest_categorical("w_steps", [4000, 8000])

        tup_params = [d_model, num_layers, num_clusters]
        if tup_params in self.list_explored_params:
            raise optuna.TrialPruned()
        else:
            self.list_explored_params.append(tup_params)

        if self.cluster == "yes":

            model = GMM(d_model=d_model, input_size=self.data_loader.input_size, num_clusters=num_clusters).to(self.device)

        else:
            d_model_gmm = 1 if self.GMM_best_model is None else self.GMM_best_model.d_model

            model = Forecasting(input_size=self.data_loader.input_size,
                                output_size=self.data_loader.output_size,
                                d_model=d_model,
                                nheads=8,
                                num_layers=num_layers,
                                attn_type=self.attn_type,
                                seed=1234,
                                device=self.device,
                                pred_len=96,
                                batch_size=self.batch_size,
                                d_model_gmm=d_model_gmm
                                ).to(self.device)

        optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)
        best_trial_valid_loss = 1e10
        for epoch in range(self.num_epochs):

            model.train()
            train_loss = 0

            for x, y in self.data_loader.train_loader:

                if self.num_optuna_run == 2:
                    x_gmm, _ = self.GMM_best_model(x.to(self.device))
                else:
                    x_gmm = None
                output, loss = model(x.to(self.device), y.to(self.device), x_gmm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step_and_update_lr()
                train_loss += loss.item()

            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            model.eval()
            valid_loss = 0

            for x, valid_y in self.data_loader.valid_loader:
                if self.num_optuna_run == 2:
                    x_gmm, _ = self.GMM_best_model(x.to(self.device))
                else:
                    x_gmm = None
                output, loss = model(x.to(self.device), valid_y.to(self.device), x_gmm)
                valid_loss += loss.item()

                if valid_loss < best_trial_valid_loss:
                    best_trial_valid_loss = valid_loss
                    if best_trial_valid_loss < self.best_overall_valid_loss:
                        self.best_overall_valid_loss = best_trial_valid_loss
                        if self.cluster == "yes":
                            self.GMM_best_model = model
                            torch.save({'model_state_dict': self.GMM_best_model.state_dict()},
                                       os.path.join(self.model_path, "{}".format(self.model_name)))
                        else:
                            self.best_model = model
                            torch.save({'model_state_dict': self.best_model.state_dict()},
                                   os.path.join(self.model_path, "{}".format(self.model_name)))

            if epoch % 5 == 0:
                print("train loss: {:.3f} epoch: {}".format(train_loss, epoch))
                print("valid loss: {:.3f}".format(valid_loss))

        return best_trial_valid_loss

    def evaluate(self):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
        self.best_model.eval()

        _, test_y = next(iter(self.data_loader.test_loader))
        total_b = len(list(iter(self.data_loader.test_loader)))

        predictions = torch.zeros(total_b, test_y.shape[0], self.pred_len)
        test_y_tot = torch.zeros(total_b, test_y.shape[0], self.pred_len)

        j = 0

        for x, test_y in self.data_loader.test_loader:
            if self.num_optuna_run == 2:
                x_gmm, _ = self.GMM_best_model(x.to(self.device))
            else:
                x_gmm = None
            output, _ = self.best_model(x=x.to(self.device), x_gmm=x_gmm)
            predictions[j] = output.squeeze(-1).cpu().detach()
            test_y_tot[j] = test_y[:, -self.pred_len:, :].squeeze(-1).cpu().detach()
            j += 1

        predictions = predictions.reshape(-1, 1)
        test_y = test_y_tot.reshape(-1, 1)

        test_loss = F.mse_loss(predictions, test_y).item()
        mse_loss = test_loss

        mae_loss = F.l1_loss(predictions, test_y).item()
        mae_loss = mae_loss

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

Train()
