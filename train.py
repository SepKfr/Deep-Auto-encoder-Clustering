import argparse
import os
import optuna
from torch import nn
from torch.optim import Adam

import dataforemater
import pandas as pd
import torch
from optuna.trial import TrialState
from clusterforecasting import ClusterForecasting
from data_loader import CustomDataLoader
from forecasting import Forecasting


class Train:
    def __init__(self):

        parser = argparse.ArgumentParser(description="train args")
        parser.add_argument("--exp_name", type=str, default="exchange")
        parser.add_argument("--model_name", type=str, default="clusterforecast")
        parser.add_argument("--num_epochs", type=int, default=50)
        parser.add_argument("--n_trials", type=int, default=2)
        parser.add_argument("--cuda", type=str, default='cuda:0')
        parser.add_argument("--pred_len", type=int, default=96)
        parser.add_argument("--max_encoder_length", type=int, default=96)
        parser.add_argument("--max_train_sample", type=int, default=128)
        parser.add_argument("--max_test_sample", type=int, default=64)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_path", type=str, default='')
        parser.add_argument('--cluster', choices=['yes', 'no'], default='no',
                            help='Enable or disable a feature (choices: yes, no)')

        args = parser.parse_args()

        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        self.exp_name = args.exp_name
        self.max_encoder_length = args.max_encoder_length

        model_dir = "forecasting_models_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model_name = "{}_{}".format(args.model_name, args.pred_len)
        self.model_path = model_dir
        self.cluster = args.cluster

        data = pd.read_csv(os.path.join("data_CP", "{}.csv".format(self.exp_name)))

        data_formatter = dataforemater.DataFormatter(args.exp_name)
        data = data[:1600]

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
        self.run_optuna(args)

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

    def run_epoch(self, trial, model, optimizer, epoch, best_trial_valid_loss):

        model.train()
        train_loss = 0

        for x_1, x_2, y in self.data_loader.train_loader:

            loss = model(x_1, x_2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        model.eval()
        valid_loss = 0

        for valid_enc, valid_dec, valid_y in self.data_loader.valid_loader:
            loss = model(valid_enc, valid_dec, valid_y)
            valid_loss += loss.item()

            if valid_loss < best_trial_valid_loss:
                best_trial_valid_loss = valid_loss
                if best_trial_valid_loss < self.best_overall_valid_loss:
                    self.best_overall_valid_loss = best_trial_valid_loss
                    self.best_model = model
                    torch.save({'model_state_dict': self.best_model.state_dict()},
                               os.path.join(self.model_path, "{}".format(self.model_name)))

        print("train loss: {:.3f} epoch: {}".format(train_loss, epoch))
        print("valid loss: {:.3f} epoch: {}".format(valid_loss, epoch))

        return best_trial_valid_loss

    def objective(self, trial):

        if self.cluster == "yes":
            model = ClusterForecasting(input_size=self.data_loader.input_size,
                                       output_size=self.data_loader.output_size,
                                       num_clusters=3,
                                       d_model=8,
                                       nheads=1,
                                       num_layers=1,
                                       attn_type="basic_attn",
                                       seed=1234,
                                       device="cpu",
                                       pred_len=96,
                                       seq_length=self.data_loader.seq_length,
                                       num_seg=self.data_loader.num_seg,
                                       batch_size=self.batch_size)
        else:
            model = Forecasting(input_size=self.data_loader.input_size,
                                output_size=self.data_loader.output_size,
                                d_model=8,
                                nheads=1,
                                num_layers=1,
                                attn_type="basic_attn",
                                seed=1234,
                                device="cpu",
                                pred_len=96,
                                batch_size=self.batch_size)

        optimizer = Adam(model.parameters())
        best_trial_valid_loss = 1e10
        for epoch in range(self.num_epochs):
            self.run_epoch(trial, model, optimizer, epoch, best_trial_valid_loss)

Train()
