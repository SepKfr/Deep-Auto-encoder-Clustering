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

        self.real_inputs = real_inputs

        self.num_features = 4
        self.device = device

        self.total_time_steps = self.max_encoder_length

        X, Y = self.create_dataloader(data, max_train_sample)

        get_num_sample = lambda l: 2 ** round(np.log2(l))

        def get_sampler(source, num_samples=-1):
            num_samples = get_num_sample(len(source)) if num_samples == -1 else num_samples
            batch_sampler = BatchSampler(
                sampler=torch.utils.data.RandomSampler(source, num_samples=num_samples),
                batch_size=self.batch_size,
                drop_last=False,
            )
            return batch_sampler

        total_batches = len(X) // self.batch_size

        self.n_folds = 3
        test_num = total_batches // self.n_folds

        all_inds = np.arange(0, len(X) - test_num * self.batch_size)

        self.hold_out_test = DataLoader(X[:test_num * self.batch_size],
                                        batch_sampler=get_sampler(X[:test_num * self.batch_size],
                                                                  max_test_sample))

        self.list_of_train_loader = []
        self.list_of_test_loader = []
        X = X[test_num * self.batch_size:]
        self.n_folds -= 1

        for i in range(self.n_folds):

            test_inds = np.arange(batch_size * test_num * i, batch_size * test_num * (i + 1))
            train_inds = list(filter(lambda x: x not in test_inds, all_inds))

            train = X[train_inds]
            test = X[test_inds]

            self.list_of_train_loader.append(DataLoader(train, batch_sampler=get_sampler(train, max_train_sample)))
            self.list_of_test_loader.append(DataLoader(test, batch_sampler=get_sampler(test, max_test_sample)))

        train_x = next(iter(self.list_of_train_loader[0]))
        self.input_size = train_x.shape[2]
        self.output_size = train_x.shape[2]

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

        if max_samples == -1:
            ranges = valid_sampling_locations
        else:
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

        return X, Y