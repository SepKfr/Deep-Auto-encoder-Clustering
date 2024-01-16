import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


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

        self.train_loader = self.create_dataloader(train_data, max_train_sample)
        self.valid_loader = self.create_dataloader(valid_data, max_test_sample)
        self.test_loader = self.create_dataloader(test_data, max_test_sample)

        train_x_1, train_x_2, train_y = next(iter(self.train_loader))
        self.input_size = train_x_2.shape[3]
        self.output_size = train_y.shape[2]
        self.seq_length = train_x_2.shape[1]
        self.num_seg = train_x_2.shape[2]

    def create_dataloader(self, data, max_samples):

        x_list = []

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

        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)
        ]
        X = torch.zeros(max_samples, self.max_encoder_length, self.num_features)
        Y = torch.zeros(max_samples, self.pred_len, self.num_features)

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced["value"].values
            algo = rpt.Pelt().fit(val)
            cp = algo.predict(pen=1)
            cp[-1] = cp[-1] - 1
            cp = torch.tensor(cp)

            change_indices = torch.where(torch.diff(cp) != 0)[0]
            tensor = torch.from_numpy(val)

            if len(change_indices) > 0:

                last_one = len(tensor) - change_indices[-1]
                change_indices = torch.cat([torch.zeros(1), change_indices])
                change_indices = torch.diff(change_indices)

                change_indices = change_indices.tolist()
                change_indices.append(last_one)
                change_indices = [int(x) for x in change_indices]
                tensors = torch.split(tensor, change_indices)
                padded_tensor = pad_sequence(tensors, padding_value=0)

                x_list.append(padded_tensor[:self.max_encoder_length, :])
                Y[i] = tensor[-self.pred_len:].unsqueeze(-1)
                X[i] = tensor[:self.max_encoder_length].unsqueeze(-1)

            else:
                Y[i] = tensor[-self.pred_len:].unsqueeze(-1)
                x_list.append(tensor[:self.max_encoder_length].unsqueeze(-1))
                X[i] = tensor[:self.max_encoder_length].unsqueeze(-1)

        max_size_1 = max(tensor.size(0) for tensor in x_list)
        max_size_2 = max(tensor.size(1) for tensor in x_list)
        tensors_final = torch.zeros(len(x_list), max_size_1, max_size_2, self.num_features)

        Y = Y[:len(x_list)]
        X = X[:len(x_list)]

        for i, tensor in enumerate(x_list):

            tensors_final[i, :tensor.shape[0], :tensor.shape[1], :] = tensor.unsqueeze(-1)

        dataset = TensorDataset(X,
                                tensors_final,
                                Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader
