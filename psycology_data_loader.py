import argparse
import pandas as pd
import sklearn.preprocessing
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler

from seed_manager import set_seed

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class PatientDataLoader:
    def __init__(self,
                 max_encoder_length,
                 max_train_sample,
                 batch_size,
                 device,
                 data,
                 seed):

        set_seed(seed)
        data = data.fillna(0.0)
        data = data.drop(['Unnamed: 0'], axis=1)
        self.n_clusters = data["apache"].nunique()

        self.real_inputs = data.columns[~data.columns.isin(['apache', 'id'])]

        data_real = data[self.real_inputs]

        data[data_real.columns] = sklearn.preprocessing.StandardScaler().fit_transform(data_real)

        self.max_encoder_length = max_encoder_length
        self.max_train_sample = max_train_sample * batch_size if max_train_sample != -1 else -1
        self.batch_size = batch_size

        self.num_features = len(data.columns) - 2
        self.device = device

        self.total_time_steps = self.max_encoder_length

        X, Y = self.create_dataloader(data, self.max_train_sample)
        permuted_indices = torch.randperm(len(X))
        X = X[permuted_indices]
        Y = Y[permuted_indices]

        len_set = len(X) // 8
        len_train = len_set * 6

        train_set_s = X[:len_train]
        train_set_l = Y[:len_train]
        valid_set_s = X[len_train:len_set + len_train]
        valid_set_l = Y[len_train:len_set + len_train]
        sample_hold_out = X[-len_set:]
        labels_hold_out = Y[-len_set:]

        train_data = TensorDataset(train_set_s, train_set_l)
        test_hold_out_data = TensorDataset(sample_hold_out, labels_hold_out)
        valid_data = TensorDataset(valid_set_s, valid_set_l)

        self.hold_out_test = DataLoader(test_hold_out_data, batch_size=self.batch_size, drop_last=True)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, drop_last=True)
        self.test_loader = DataLoader(valid_data, batch_size=self.batch_size, drop_last=True)

        test_x, test_y = next(iter(self.hold_out_test))
        self.input_size = test_x.shape[2]
        self.output_size = test_x.shape[2]
        self.len_train = len(self.train_loader)
        self.len_test = len(self.test_loader)

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
            cluster_id = torch.from_numpy(sliced["apache"].values).unsqueeze(-1)
            Y[i] = cluster_id
            X[i] = tensor

        return X, Y



