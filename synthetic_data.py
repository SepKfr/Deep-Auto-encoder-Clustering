import math
import torch
import gpytorch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, BatchSampler

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class SyntheticDataLoader:

    def __init__(self, batch_size, max_samples):

        samples, labels = self.get_synthetic_samples()
        len_samples = len(samples)
        permuted_indices = torch.randperm(len_samples)
        samples = samples[permuted_indices]
        labels = labels[permuted_indices]

        if max_samples != -1:

            samples = samples[:max_samples*batch_size]
            labels = labels[:max_samples*batch_size]

        len_set = len(samples) // 8
        len_train = len_set * 6

        train_set_s = samples[:len_train]
        train_set_l = labels[:len_train]
        valid_set_s = samples[len_train:len_set+len_train]
        valid_set_l = labels[len_train:len_set+len_train]
        sample_hold_out = samples[-len_set:]
        labels_hold_out = labels[-len_set:]

        hold_out_tensor_data = TensorDataset(sample_hold_out, labels_hold_out)

        self.hold_out_test = DataLoader(hold_out_tensor_data, batch_size=batch_size, drop_last=True)

        self.list_of_test_loader = []
        self.list_of_train_loader = []

        train_data = TensorDataset(train_set_s, train_set_l)

        self.list_of_train_loader.append(DataLoader(train_data, batch_size=batch_size, drop_last=True))
        self.list_of_test_loader.append(DataLoader(TensorDataset(valid_set_s,
                                                                 valid_set_l),
                                                                 batch_size=batch_size,
                                                                 drop_last=True))
        self.n_folds = 1
        self.input_size = 1
        self.output_size = 1
        self.len_train = len(self.list_of_train_loader[0])
        self.len_test = len(self.list_of_test_loader[0])

    def get_synthetic_samples(self):

        # Training data is 100 points in [0,1] inclusive regularly spaced
        train_x = torch.linspace(0, 1, 100).view(1, -1, 1).repeat(4, 1, 1)
        # True functions are sin(2pi x), cos(2pi x), sin(pi x), cos(pi x)
        sin_y = torch.sin(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
        sin_y_short = torch.sin(train_x[0] * math.pi) + 0.5 * torch.rand(1, 100, 1)
        cos_y = torch.cos(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
        cos_y_short = torch.cos(train_x[0] * math.pi) + 0.5 * torch.rand(1, 100, 1)
        train_y = torch.cat((sin_y, sin_y_short, cos_y, cos_y_short)).squeeze(-1)

        # We will use the simplest form of GP model, exact inference

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([4]))
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(batch_shape=torch.Size([4])),
                    batch_shape=torch.Size([4])
                )

            def forward(self, x):

                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([4]))
        model = ExactGPModel(train_x, train_y, likelihood)

        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iter = 100

        print("Fitting GP...")
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y).sum()
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

        with torch.no_grad():

            test_x = torch.linspace(0, 1, 100).view(1, -1, 1).repeat(4, 1, 1)

            observed_pred = likelihood(model(test_x)).sample_n(10240)
            # Get mean
            samples = observed_pred.detach().cpu()
            samples = samples.reshape(10240*4, 100, 1)
            label = torch.tensor([0, 1, 2, 3]*10240)
            labels = label.unsqueeze(-1).repeat(1, 100)
            labels = labels.reshape(samples.shape)

        return samples, labels



