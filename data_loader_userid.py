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
        permuted_indices = torch.randperm(len(X))
        X = X[permuted_indices]
        Y = Y[permuted_indices]

        get_num_sample = lambda l: 2 ** round(np.log2(l))

        def get_sampler(source, num_samples=-1):
            num_samples = get_num_sample(len(source)) if num_samples == -1 else num_samples
            batch_sampler = BatchSampler(
                sampler=torch.utils.data.RandomSampler(source, num_samples=num_samples),
                batch_size=self.batch_size,
                drop_last=False,
            )
            return batch_sampler

        self.n_folds = 1
        test_num = int(len(X) * 0.1)

        all_inds = np.arange(0, len(X) - test_num)

        hold_out_dataset = TensorDataset(X[:test_num],
                                         Y[:test_num])

        self.hold_out_test = DataLoader(hold_out_dataset,
                                        batch_sampler=get_sampler(hold_out_dataset,
                                                                  max_test_sample))

        self.list_of_train_loader = []
        self.list_of_test_loader = []
        X = X[test_num:]

        for i in range(self.n_folds):

            test_inds = np.arange(test_num * i, test_num * (i + 1))
            train_inds = list(filter(lambda x: x not in test_inds, all_inds))

            train_x = X[train_inds]
            train_y = Y[train_inds]
            test_x = X[test_inds]
            test_y = Y[test_inds]

            train_data = TensorDataset(train_x, train_y)
            valid_data = TensorDataset(test_x, test_y)

            self.list_of_train_loader.append(DataLoader(train_data, batch_sampler=get_sampler(train_data, max_train_sample)))
            self.list_of_test_loader.append(DataLoader(valid_data, batch_sampler=get_sampler(valid_data, max_test_sample)))

        train_x, _ = next(iter(self.list_of_train_loader[0]))
        self.input_size = train_x.shape[2]
        self.output_size = train_x.shape[2]
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

        if max_samples == -1:
            ranges = valid_sampling_locations
        else:
            ranges = [valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)]

        X = torch.zeros(max_samples, self.total_time_steps, self.num_features)
        Y = torch.zeros(max_samples, 1)

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced[self.real_inputs].values
            tensor = torch.tensor(val)
            Y[i] = identifier
            X[i] = tensor

        return X, Y