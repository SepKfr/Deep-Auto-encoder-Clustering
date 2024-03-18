import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from scipy.stats import ks_2samp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from itertools import accumulate

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

        real_inputs = real_inputs[0]
        self.real_inputs = real_inputs

        self.num_features = 2
        self.device = device

        X = self.create_dataloader(data, max_train_sample)

        total_batches = int(len(X) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.8)
        valid_len = int(total_batches * batch_size * 0.1)
        test_len = int(total_batches * batch_size * 0.1)

        train = X[:train_len]
        valid = X[train_len:train_len + valid_len]
        test = X[train_len + valid_len:train_len + valid_len + test_len]

        get_num_sample = lambda l: 2 ** round(np.log2(l))

        def get_sampler(source, num_samples=-1):
            num_samples = get_num_sample(len(source)) if num_samples == -1 else num_samples
            batch_sampler = BatchSampler(
                sampler=torch.utils.data.RandomSampler(source, num_samples=num_samples),
                batch_size=self.batch_size,
                drop_last=False,
            )
            return batch_sampler

        self.train_loader = DataLoader(train, batch_sampler=get_sampler(train, max_train_sample))
        self.valid_loader = DataLoader(valid, batch_sampler=get_sampler(valid, max_test_sample))
        self.test_loader = DataLoader(test, batch_sampler=get_sampler(test, max_test_sample))

        train_x = next(iter(self.train_loader))
        self.input_size = train_x.shape[2]
        self.output_size = train_x.shape[2]

    def create_dataloader(self, data, max_samples):

        def detect_significant_events_ma(data, covar, window_size, threshold_factor):

            moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
            moving_avg2 = np.convolve(moving_avg, np.ones(window_size) / window_size, mode='valid')
            residuals = np.abs(moving_avg[window_size - 1:] - moving_avg2)

            # Calculate the standard deviation of the residuals
            residual_std = np.std(residuals)

            # Set the threshold as a multiple of the standard deviation
            threshold = threshold_factor * residual_std

            # Find indices of significant events where residuals exceed the threshold
            significant_events = np.where(residuals > threshold)[0]
            significant_events = torch.tensor(significant_events)

            cp = torch.where(torch.roll(significant_events, shifts=-1) - significant_events > 1, torch.tensor(1),
                             torch.tensor(0))
            change_point_indices = torch.nonzero(cp).squeeze()

            # Add the first and last indices to capture the entire range
            change_point_indices = torch.cat(
                (torch.tensor([0]), change_point_indices, torch.tensor([len(significant_events)])))

            # Calculate split sizes
            split_sizes = change_point_indices[1:] - change_point_indices[:-1]

            # Filter out zero-sized splits (if any)
            non_zero_splits = split_sizes[split_sizes != 0]

            # Compute the starting indices for each split
            split_starts = torch.cumsum(non_zero_splits, dim=0)

            # Split the tensor based on change points, excluding the last index of each chunk
            split_tensors = [significant_events[start+1:end+1] if start > change_point_indices[0] else
                             significant_events[start:end+1] for start, end in
                             zip(change_point_indices[:-1], split_starts)]

            return split_tensors, data, covar

        total_ind = []
        total_tensors_q = []
        total_tensors_c = []
        list_of_lens = []
        for identifier, df in data.groupby("id"):

            val = df["Q"].values
            covar = df["Conductivity"].values
            list_of_inner, trg, cov = detect_significant_events_ma(val, covar, window_size=7, threshold_factor=3)
            list_of_lens.append(len(list_of_inner))
            total_ind.append(list_of_inner)
            total_tensors_q.append(trg)
            total_tensors_c.append(cov)

        total_ind = [item for sublist in total_ind for item in sublist]

        max_length = max(len(t) for t in total_ind)

        tot_chunk_q = []
        tot_chunk_c = []
        cumulative_sum = list(accumulate(list_of_lens))

        def get_index(ind):
            for j, x in enumerate(cumulative_sum):
                if x - ind > 0:
                    return j

        for i, tensor_chunk in enumerate(total_ind):

            num_to_add = (max_length - len(tensor_chunk)) // 2

            # Get the index range for expanding the chunk
            start_index = max(0, tensor_chunk[0] - num_to_add)
            end_index = min(len(data), tensor_chunk[-1] + num_to_add + 1)

            tgt = total_tensors_q[get_index(i)]
            cov = total_tensors_c[get_index(i)]

            q = torch.tensor(tgt[start_index:end_index])
            c = torch.tensor(cov[start_index:end_index])
            tot_chunk_q.append(q)
            tot_chunk_c.append(c)

        padded_tensor_q = pad_sequence(tot_chunk_q, padding_value=0)
        padded_tensor_c = pad_sequence(tot_chunk_c, padding_value=0)
        padded_tensor_q = padded_tensor_q.permute(1, 0).unsqueeze(-1)
        padded_tensor_c = padded_tensor_c.permute(1, 0).unsqueeze(-1)

        X = torch.cat([padded_tensor_q, padded_tensor_c], dim=-1).to(torch.float)
        return X