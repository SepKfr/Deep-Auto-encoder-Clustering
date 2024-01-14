import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, TensorDataset, DataLoader
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
        self.max_train_sample = max_train_sample
        self.max_test_sample = max_test_sample
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

        self.num_features = 2
        self.device = device

        train_data = pd.DataFrame(
            dict(
                value=train[target_col],
                reals=train[real_inputs] if len(real_inputs) > 0 else None,
                cat=train["CP"],
                group=train["id"],
                time_idx=np.arange(train_len),
            )
        )

        valid_data = pd.DataFrame(
            dict(
                value=valid[target_col],
                reals=valid[real_inputs] if len(real_inputs) > 0 else None,
                cat=valid["CP"],
                group=valid["id"],
                time_idx=np.arange(train_len, train_len+valid_len),
            )
        )

        test_data = pd.DataFrame(
            dict(
                value=test[target_col],
                reals=test[real_inputs] if len(real_inputs) > 0 else None,
                cat=test["CP"],
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

    def create_dataloader(self, data, sample):

        num_samples = len(data)

        X = torch.zeros(num_samples, self.max_encoder_length, self.num_features)
        Y = torch.zeros(num_samples, self.pred_len, 1)
        i = 0
        x_list = []
        x_reals = None

        for id, df in data.groupby("group"):

            for start in range(len(df)):

                x = df[start:start+self.max_encoder_length]
                y = df[start+self.max_encoder_length: start+self.total_time_steps]
                x_length = len(x)
                y_length = len(y)

                if x_length + y_length == self.total_time_steps:

                    x_value = torch.from_numpy(x["value"].values).unsqueeze(-1)
                    x_cat = torch.from_numpy(x["cat"].values)

                    if len(self.real_inputs) > 0:
                        x_reals = torch.from_numpy(x["reals"].values)
                        if len(self.real_inputs) == 1:
                            x_reals = x_reals.unsqueeze(-1)

                    y = torch.from_numpy(y["value"].values).unsqueeze(-1)

                    change_indices = torch.where(torch.diff(x_cat) != 0)[0]

                    x_cat = x_cat.unsqueeze(-1)
                    if x_reals is not None:
                        x_total = [x_value, x_reals, x_cat]
                    else:
                        x_total = [x_value, x_cat]

                    x_df = torch.cat(x_total, dim=-1)

                    if len(change_indices) > 0:

                        last_one = len(x_cat) - change_indices[-1]
                        change_indices = torch.cat([torch.zeros(1), change_indices])
                        change_indices = torch.diff(change_indices)
                        change_indices = change_indices.tolist()
                        change_indices.append(last_one)
                        change_indices = [int(x) for x in change_indices]

                        tensors = torch.split(x_df, change_indices)
                        padded_tensor = pad_sequence(tensors, padding_value=0)

                        x_list.append(padded_tensor)
                        Y[i] = y
                        X[i] = x_df

        max_size_1 = max(tensor.size(0) for tensor in x_list)
        max_size_2 = max(tensor.size(1) for tensor in x_list)
        tensors_final = torch.zeros(len(x_list), max_size_1, max_size_2, self.num_features)
        Y = Y[:len(x_list)]
        X = X[:len(x_list)]

        for i, tensor in enumerate(x_list):
            tensors_final[i, :tensor.shape[0], :tensor.shape[1], :] = tensor

        dataset = TensorDataset(X, tensors_final, Y)
        dataloader = DataLoader(dataset, batch_sampler=BatchSampler(range(sample),
                                                                    batch_size=self.batch_size,
                                                                    drop_last=False))


        return dataloader