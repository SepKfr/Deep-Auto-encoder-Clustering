import random
import numpy as np
import pandas as pd
import torch
import ruptures as rpt
from scipy.stats import ks_2samp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import find_peaks
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

        dataset = pd.DataFrame(
            dict(
                value=data[target_col],
                reals=data[real_inputs] if len(real_inputs) > 0 else None,
                group=data["id"],
                time_idx=np.arange(len(data)),
            )
        )

        X = self.create_dataloader(dataset, max_train_sample)

        total_batches = int(len(X) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.6)
        valid_len = int(total_batches * batch_size * 0.2)
        test_len = int(total_batches * batch_size * 0.2)

        train = X[:train_len]
        valid = X[train_len:train_len + valid_len]
        test = X[train_len + valid_len:train_len + valid_len + test_len]

        def get_sampler(source, num_samples):
            batch_sampler = BatchSampler(
                sampler=torch.utils.data.RandomSampler(source, num_samples=num_samples),
                batch_size=self.batch_size,
                drop_last=True,
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
            residuals = np.abs(data[window_size - 1:] - moving_avg)

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

            expansion_percentage = 0.10

            # Filter out chunks with less than 10 data points and expand others
            expanded_chunks = []
            expanded_chunks_covar = []
            for tensor_chunk in split_tensors:
                if len(tensor_chunk) < 10:
                    continue  # Skip chunks with less than 10 data points
                else:
                    # Calculate the number of data points to add before and after
                    num_to_add = int(len(tensor_chunk) * expansion_percentage)

                    # Get the index range for expanding the chunk
                    start_index = max(0, tensor_chunk[0] - num_to_add)
                    end_index = min(len(data), tensor_chunk[-1] + num_to_add + 1)

                    chunk_to_add = torch.from_numpy(data[start_index:end_index])
                    if covar is not None:

                        expanded_chunks_covar.append(torch.from_numpy(covar[start_index:end_index]))

                    expanded_chunks.append(chunk_to_add)

            return expanded_chunks, expanded_chunks_covar, threshold

        total_tensors_q = []
        total_tensors_c = []
        for identifier, df in data.groupby("group"):
            val = df["value"].values
            covar = df["reals"].values
            list_of_events_q, list_of_events_c, _ = detect_significant_events_ma(val, covar, window_size=30,
                                                                                 threshold_factor=3)
            padded_tensor_q = pad_sequence(list_of_events_q, padding_value=0)
            padded_tensor_c = pad_sequence(list_of_events_c, padding_value=0)
            total_tensors_q.append(padded_tensor_q)
            total_tensors_c.append(padded_tensor_c)

        max_size_1 = max(tensor.size(0) for tensor in total_tensors_q)
        max_size_2 = max(tensor.size(1) for tensor in total_tensors_q)
        tensors_final = torch.zeros(len(total_tensors_q), max_size_1, max_size_2, self.num_features)
        for i in range(len(total_tensors_q)):

            q = total_tensors_q[i]
            c = total_tensors_c[i]
            tensors_final[i, :q.shape[0], :q.shape[1], 0] = q
            tensors_final[i, :q.shape[0], :q.shape[1], 1] = c

        X = tensors_final.reshape(-1, tensors_final.shape[2], self.num_features)
        return X