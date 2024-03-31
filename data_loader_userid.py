import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from scipy.stats import ks_2samp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class UserDataLoader:
    def __init__(self,
                 max_encoder_length,
                 max_train_sample,
                 batch_size,
                 device,
                 data,
                 target_col,
                 real_inputs):

        self.max_encoder_length = max_encoder_length
        self.max_train_sample = max_train_sample * batch_size if max_train_sample != -1 else -1
        self.batch_size = batch_size

        self.real_inputs = real_inputs

        self.num_features = 4
        self.device = device

        self.total_time_steps = self.max_encoder_length

        X, Y = self.create_dataloader(data, self.max_train_sample)
        permuted_indices = torch.randperm(len(X))
        X = X[permuted_indices]
        Y = Y[permuted_indices]

        len_set = len(X) // 8
        len_train = len_set * 6
        len_set *= 2

        train_set_s = X[:len_train]
        train_set_l = Y[:len_train]
        valid_set_s = X[len_train:len_set + len_train]
        valid_set_l = Y[len_train:len_set + len_train]
        sample_hold_out = X[-len_set:]
        labels_hold_out = Y[-len_set:]

        self.list_of_test_loader = []
        self.list_of_train_loader = []

        train_data = TensorDataset(train_set_s, train_set_l)
        test_hold_out_data = TensorDataset(sample_hold_out, labels_hold_out)
        valid_data = TensorDataset(valid_set_s, valid_set_l)

        self.hold_out_test = DataLoader(test_hold_out_data, batch_size=self.batch_size, drop_last=True)
        self.list_of_train_loader.append(DataLoader(train_data, batch_size=self.batch_size, drop_last=True))
        self.list_of_test_loader.append(DataLoader(valid_data, batch_size=self.batch_size, drop_last=True))

        self.n_folds = 1
        test_x, _ = next(iter(self.hold_out_test))
        self.input_size = test_x.shape[2]
        self.output_size = test_x.shape[2]
        self.len_train = len(self.list_of_train_loader[0])
        self.len_test = len(self.list_of_test_loader[0])

    def create_dataloader(self, data, max_samples):

        valid_sampling_locations, split_data_map = zip(
            *[
                (
                    (identifier, self.total_time_steps + i),
                    (identifier, df)
                )
                for identifier, df in data.groupby("id")
                if (num_entries := len(df)) >= self.total_time_steps
                for i in range(num_entries - self.total_time_steps + 1)
            ]
        )
        valid_sampling_locations = list(valid_sampling_locations)
        split_data_map = dict(split_data_map)

        max_samples = len(valid_sampling_locations) if max_samples == -1 else max_samples

        ranges = [valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)]

        X = torch.zeros(max_samples, self.total_time_steps, self.num_features)
        Y = torch.zeros(max_samples, self.total_time_steps, 1)

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced[self.real_inputs].values
            tensor = torch.tensor(val)
            cluster_id = torch.zeros(self.total_time_steps).fill_(identifier).unsqueeze(-1)
            Y[i] = cluster_id
            X[i] = tensor

        return X, Y