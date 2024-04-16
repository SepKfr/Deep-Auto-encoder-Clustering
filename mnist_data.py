import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

train = torchvision.datasets.MNIST('./', download=True)
train_data = train.data.detach().numpy()
labels = train.targets.detach().numpy()

train = train_data.reshape(-1, 28, 28, 1)

data_val = train[45000:]
labels_val = labels[45000:]
data_train = train[:45000]
labels_train = labels[:45000]


class MnistDataLoader:
    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.train_loader, self.test_loader = self.get_dataloader()
        self.hold_out_test = self.test_loader

        test_x, _ = next(iter(self.hold_out_test))
        self.input_size = test_x.shape[2] * test_x.shape[3]
        self.output_size = test_x.shape[2] * test_x.shape[3]
        self.len_train = len(self.train_loader)
        self.len_test = len(self.test_loader)


    @staticmethod
    def interpolate_arrays(arr1, arr2, num_steps=100, interpolation_length=0.3):
        """Interpolates linearly between two arrays over a given number of steps.
        The actual interpolation happens only across a fraction of those steps.

        Args:
            arr1 (np.array): The starting array for the interpolation.
            arr2 (np.array): The end array for the interpolation.
            num_steps (int): The length of the interpolation array along the newly created axis (default: 100).
            interpolation_length (float): The fraction of the steps across which the actual interpolation happens (default: 0.3).

        Returns:
            np.array: The final interpolated array of shape ([num_steps] + arr1.shape).
        """
        assert arr1.shape == arr2.shape, "The two arrays have to be of the same shape"
        start_steps = int(num_steps*interpolation_length)
        inter_steps = int(num_steps*((1-interpolation_length)/2))
        end_steps = num_steps - start_steps - inter_steps
        interpolation = np.zeros([inter_steps]+list(arr1.shape))
        arr_diff = arr2 - arr1
        for i in range(inter_steps):
            interpolation[i] = arr1 + (i/(inter_steps-1))*arr_diff
        start_arrays = np.concatenate([np.expand_dims(arr1, 0)] * start_steps)
        end_arrays = np.concatenate([np.expand_dims(arr2, 0)] * end_steps)
        final_array = np.concatenate((start_arrays, interpolation, end_arrays))
        return final_array

    def get_data_generator(self, time_series=True):
        """Creates a data generator for the training.

        Args:
            time_series (bool): Indicates whether or not we want interpolated MNIST time series or just
                normal MNIST batches.

        Returns:
            generator: Data generator for the batches."""

        def batch_generator(mode="train", time_steps=100):
            """Generator for the data batches.

            Args:
                mode (str): Mode in ['train', 'val'] that decides which data set the generator
                    samples from (default: 'train').
                batch_size (int): The size of the batches (default: 100).

            Yields:
                np.array: Data batch.
            """
            assert mode in ["train", "val"], "The mode should be in {train, val}."
            if mode == "train":
                images = data_train.copy()
                labels = labels_train.copy()
            elif mode == "val":
                images = data_val.copy()
                labels = labels_val.copy()

            while True:
                indices = np.random.permutation(np.arange(len(images)))
                images = images[indices]
                labels = torch.from_numpy(labels[indices]).to(torch.float)
                labels = labels.unsqueeze(-1).repeat(1, time_steps).unsqueeze(-1)

                for i, image in enumerate(images):
                    start_image = image
                    end_image = images[np.random.choice(np.where(labels == (labels[i] + 1) % 10)[0])]
                    interpolation = self.interpolate_arrays(start_image, end_image)
                    inter_p = interpolation + np.random.normal(scale=0.01, size=interpolation.shape)
                    inter_p = torch.from_numpy(inter_p).to(torch.float)
                    yield inter_p, labels[i]

        return batch_generator


    def get_dataloader(self):

        data_generator = self.get_data_generator()

        train_gen = data_generator("train")
        val_gen = data_generator("val")

        time_steps = 100

        num_iter = 1000

        def get_xy(mode='train'):

            list_x = []
            list_y = []

            if mode == "train":
                gen = train_gen
            else:
                gen = val_gen

            for i in range(num_iter):
                x, y = next(gen)
                list_x.append(x)
                list_y.append(y)

            X = torch.cat(list_x, dim=0)
            Y = torch.cat(list_y, dim=0)
            X = X.reshape(num_iter, time_steps, 28, 28)
            Y = Y.reshape(num_iter, time_steps, 1)
            return X, Y

        X_train, Y_train = get_xy()
        X_valid, Y_valid = get_xy('valid')
        tensor_dataset_train = TensorDataset(X_train, Y_train)
        tensor_dataset_valid = TensorDataset(X_valid, Y_valid)
        data_loader_train = DataLoader(tensor_dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_loader_valid = DataLoader(tensor_dataset_valid, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return data_loader_train, data_loader_valid












