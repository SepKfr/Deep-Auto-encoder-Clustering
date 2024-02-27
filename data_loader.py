import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from scipy.stats import ks_2samp
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class CustomDataLoader:
    def __init__(self,
                 max_encoder_length,
                 pred_len,
                 max_train_sample,
                 max_test_sample,
                 batch_size,
                 device,
                 data,
                 target_col,
                 real_inputs):

        self.max_encoder_length = max_encoder_length
        self.pred_len = pred_len
        self.max_train_sample = max_train_sample * batch_size
        self.max_test_sample = max_test_sample * batch_size
        self.batch_size = batch_size
        seed = 1234
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        total_batches = int(len(data) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.6)
        valid_len = int(total_batches * batch_size * 0.2)
        test_len = int(total_batches * batch_size * 0.2)

        train = data[:train_len]
        valid = data[train_len:train_len+valid_len]
        test = data[train_len+valid_len:train_len+valid_len+test_len]
        self.real_inputs = real_inputs

        self.num_features = 1
        self.device = device

        train_data = pd.DataFrame(
            dict(
                value=train[target_col],
                reals=train[real_inputs] if len(real_inputs) > 0 else None,
                group=train["id"],
                time_idx=np.arange(train_len),
            )
        )

        valid_data = pd.DataFrame(
            dict(
                value=valid[target_col],
                reals=valid[real_inputs] if len(real_inputs) > 0 else None,
                group=valid["id"],
                time_idx=np.arange(train_len, train_len+valid_len),
            )
        )

        test_data = pd.DataFrame(
            dict(
                value=test[target_col],
                reals=test[real_inputs] if len(real_inputs) > 0 else None,
                group=test["id"],
                time_idx=np.arange(train_len+valid_len, train_len+valid_len+test_len),
            )
        )
        self.total_time_steps = self.max_encoder_length + self.pred_len

        self.train_loader, train_unique = self.create_dataloader(train_data, max_train_sample)
        self.valid_loader, valid_unique = self.create_dataloader(valid_data, max_test_sample)
        self.test_loader, test_unique = self.create_dataloader(test_data, max_test_sample)

        train_x, train_y = next(iter(self.train_loader))
        self.input_size = train_x.shape[2]
        self.output_size = train_y.shape[2]
        self.n_uniques = max(train_unique, valid_unique, test_unique)

    def create_dataloader(self, data, max_samples):

        valid_sampling_locations, split_data_map = zip(
            *[
                (
                    (identifier, self.total_time_steps + i),
                    (identifier, df)
                )
                for identifier, df in data.groupby("group")
                if (num_entries := len(df)) >= self.total_time_steps
                for i in range(num_entries - self.total_time_steps + 1)
            ]
        )
        valid_sampling_locations = list(valid_sampling_locations)
        split_data_map = dict(split_data_map)

        ranges = [valid_sampling_locations[i] for i in np.random.choice(
                  len(valid_sampling_locations), max_samples, replace=False)]

        X = torch.zeros(max_samples, self.max_encoder_length, self.num_features+1)
        Y = torch.zeros(max_samples, self.pred_len, self.num_features)
        n_uniques = []

        def detect_change_points_distribution_shift(time_series_data, window_size=9):
            """
            Detect change points in a time series using distribution shift detection.

            Args:
            - time_series_data (numpy array): The time series data.
            - window_size (int): The size of the window for analyzing distribution shift.

            Returns:
            - change_points (list): List of detected change points.
            """
            change_points = []

            # Calculate the number of windows
            num_windows = len(time_series_data) // window_size

            # Iterate through windows
            for i in range(num_windows):
                start_index = i * window_size
                end_index = start_index + window_size

                try:
                    # Split the time series into two segments: before and after the change point
                    segment_before = time_series_data[:end_index]
                    segment_after = time_series_data[end_index:]

                    # Compute Kolmogorov-Smirnov statistic to measure distribution shift
                    ks_statistic, _ = ks_2samp(segment_before, segment_after)

                    # Threshold for detecting significant distribution shift
                    threshold = 0.05  # Adjust as needed based on significance level

                    # Check if distribution shift is significant
                    if ks_statistic > threshold:
                        change_points.append(end_index)
                except ValueError:
                    pass

            return change_points

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced["value"].values
            cp = detect_change_points_distribution_shift(val)
            cp[-1] = cp[-1] - 1
            cp = torch.tensor(cp)
            val = torch.tensor(val)

            one_hot_encoding = torch.zeros(len(sliced))
            seq_of_cp = torch.cumsum(one_hot_encoding.scatter_(0, cp, 1), dim=0)
            n_uniques.append(len(torch.unique(seq_of_cp)))

            val = torch.cat([val.unsqueeze(-1), seq_of_cp.unsqueeze(-1)], dim=-1)
            X[i] = val[:self.max_encoder_length, :]
            Y[i] = val[-self.pred_len:, 0:1]

        n_unique = max(n_uniques)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader, n_unique
