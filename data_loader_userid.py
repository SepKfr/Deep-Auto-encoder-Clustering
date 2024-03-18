import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from scipy.stats import ks_2samp
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class UserDataLoader:
    def __init__(self,
                 max_encoder_length,
                 max_train_sample,
                 max_test_sample,
                 batch_size,
                 device,
                 data,
                 target_col,
                 real_inputs):

        self.max_encoder_length = max_encoder_length
        self.max_train_sample = max_train_sample * batch_size
        self.max_test_sample = max_test_sample * batch_size
        self.batch_size = batch_size

        total_batches = int(len(data) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.6)
        valid_len = int(total_batches * batch_size * 0.2)
        test_len = int(total_batches * batch_size * 0.2)

        train = data[:train_len]
        valid = data[train_len:train_len+valid_len]
        test = data[train_len+valid_len:train_len+valid_len+test_len]
        self.real_inputs = real_inputs

        self.num_features = 4
        self.device = device

        self.total_time_steps = self.max_encoder_length

        self.train_loader = self.create_dataloader(train, max_train_sample)
        self.valid_loader = self.create_dataloader(valid, max_test_sample)
        self.test_loader = self.create_dataloader(test, max_test_sample)

        train_x, train_y = next(iter(self.train_loader))
        self.input_size = train_x.shape[2]
        self.output_size = train_y.shape[1]

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

        ranges = [valid_sampling_locations[i] for i in np.random.choice(
                  len(valid_sampling_locations), max_samples, replace=False)]

        X = torch.zeros(max_samples, self.total_time_steps, self.num_features+1)
        Y = torch.zeros(max_samples, self.total_time_steps, 1)

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced[self.real_inputs].values
            tensor = torch.tensor(val)
            cluster_id = torch.zeros(self.total_time_steps).fill_(identifier).unsqueeze(-1)
            final_tensor = torch.cat([tensor, cluster_id], dim=-1)
            Y[i] = cluster_id
            X[i] = final_tensor

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return dataloader