import dataforemater
import pandas as pd
import torch
import numpy as np
from itertools import accumulate
import torch.nn as nn


class CustomDataLoaderForecasting:
    def __init__(self,
                 max_encoder_length,
                 pred_len,
                 max_train_sample,
                 max_test_sample,
                 batch_size,
                 data,
                 target_col,
                 real_inputs):

        self.max_encoder_length = max_encoder_length
        self.pred_len = pred_len
        self.max_train_sample = max_train_sample * batch_size
        self.max_test_sample = max_test_sample * batch_size
        self.batch_size = batch_size
        self.num_features = 2
        self.total_time_steps = max_encoder_length + pred_len

        real_inputs = real_inputs[0]
        self.real_inputs = real_inputs
        self.create_dataloader(data, max_train_sample)

    def detect_significant_events_ma(self, data, window_size, threshold_factor):

        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        padding = (window_size - 1) // 2
        mv_avg = nn.AvgPool1d(kernel_size=window_size, padding=padding, stride=1)(data)
        data = data.reshape(-1)
        mv_avg = mv_avg.reshape(-1)
        residuals = torch.abs(data - mv_avg)

        # Calculate the standard deviation of the residuals
        residual_std = torch.std(residuals)

        # Set the threshold as a multiple of the standard deviation
        threshold = threshold_factor * residual_std

        # Find indices of significant events where residuals exceed the threshold
        change_indices = torch.where(residuals > threshold)[0]

        cp = torch.where(torch.diff(change_indices) > 1, torch.tensor(1),
                         torch.tensor(0))

        cp = torch.nonzero(cp)

        if len(cp) >= 1:
            cp = torch.tensor(cp).reshape(-1)
        else:
            cp = torch.tensor([])

        tensor = data

        if len(cp) > 0:
            last_one = len(tensor) - cp[-1]
            change_indices = torch.cat([torch.zeros(1), cp])
            change_indices = torch.diff(change_indices)

            change_indices = change_indices.tolist()
            change_indices.append(last_one)
            change_indices = [int(x) for x in change_indices]
            tensors = torch.split(tensor, change_indices)
            print(change_indices)
            print(tensors)

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

        X = torch.zeros(max_samples, self.total_time_steps, self.num_features)
        Y = torch.zeros(max_samples, self.total_time_steps, self.num_features)

        list_of_lens = []
        total_ind = []
        total_tensors = []

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = sliced[self.real_inputs].values
            split_tensors = self.detect_significant_events_ma(val, window_size=7, threshold_factor=1)
            if split_tensors is not None:
                list_of_lens.append(len(split_tensors))
                total_ind.append(split_tensors)
                total_tensors.append(val)

        total_ind = [item for sublist in total_ind for item in sublist]

        max_length = max(len(t) for t in total_ind)

        tot_chunks = []
        cumulative_sum = list(accumulate(list_of_lens))

        def get_index(ind):
            for j, x in enumerate(cumulative_sum):
                if x - ind > 0:
                    return j

        for i, tensor_chunk in enumerate(total_ind):

            print(tensor_chunk)
            if len(tensor_chunk) >= 1:

                tgt_check = total_tensors[get_index(i)][tensor_chunk[0]:tensor_chunk[-1]+1]

                max_id = np.argmax(tgt_check)
                inds = torch.arange(tensor_chunk[0].item(), tensor_chunk[-1].item()+1)

                num_to_add = max_length // 2

                # Get the index range for expanding the chunk
                start_index = max(0, inds[max_id] - num_to_add)
                end_index = min(len(data), inds[max_id] + num_to_add + 1)

                tgt = total_tensors[get_index(i)]

                i = 0
                while i <= 100:

                    try:
                        tgt = torch.tensor(tgt[start_index:end_index])

                        diff = np.argmax(tgt) - num_to_add

                        if abs(diff) > 5:

                            if diff < 0:
                                start_index = max(0, start_index-5)
                                end_index = max(0, end_index-5)
                            else:
                                start_index = min(len(data), start_index + 5)
                                end_index = min(len(data), end_index + 5)
                            i += 1
                        else:

                            tot_chunks.append(tgt)

                            for j in range(4):
                                s = min(len(data), start_index + j)
                                e = min(len(data), end_index + j)
                                tot_chunks.append(torch.tensor(tgt[s:e]))

                                s = max(0, start_index - j)
                                e = max(0, end_index - j)
                                tot_chunks.append(torch.tensor(tgt[s:e]))

                            break
                    except ValueError:
                        break



def main():
    exp_name = "watershed"
    data_formatter = dataforemater.DataFormatter(exp_name)
    data_path = "{}.csv".format(exp_name)
    df = pd.read_csv(data_path, dtype={'date': str})
    df.sort_values(by=["id", "hours_from_start"], inplace=True)
    data = data_formatter.transform_data(df)

    data_loader = CustomDataLoaderForecasting(real_inputs=data_formatter.real_inputs,
                                              max_encoder_length=192,
                                              pred_len=24,
                                              max_train_sample=32,
                                              max_test_sample=32,
                                              batch_size=32,
                                              data=data,
                                              target_col=data_formatter.target_column)


if __name__ == '__main__':
    main()
