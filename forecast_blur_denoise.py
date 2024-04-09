import numpy as np
import random
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import MeanFieldVariationalDistribution, VariationalStrategy


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32, mean_type='linear'):

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPp(DeepGP):
    def __init__(self, num_hidden_dims, num_inducing):

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=1,
            mean_type='constant',
            num_inducing=num_inducing
        )

        super().__init__()

        self.num_hidden_dims = num_hidden_dims
        self.hidden_layer = hidden_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=1)

    def forward(self, inputs):

        dist = self.hidden_layer(inputs)
        return dist

    def predict(self, x):

        preds = torch.zeros_like(x)
        dist = self(x)
        for i in range(self.num_hidden_dims):
            pred = self.likelihood(dist)
            pred_mean = pred.mean.mean(0)
            preds[:, :, i] = pred_mean.squeeze(-1)

        return preds, dist


class BlurDenoiseModel(nn.Module):
    def __init__(self, model, d_model, num_inducing, gp, no_noise=False, iso=False):
        """
        Blur and Denoise model.

        Args:
        - model (nn.Module): Underlying forecasting model for adding and removing noise.
        - d_model (int): Dimensionality of the model.
        - num_inducing (int): Number of inducing points for the GP model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during
          denoising (denoise predictions directly).
        - iso (bool): Flag indicating whether to use isotropic noise.
        """
        super(BlurDenoiseModel, self).__init__()

        self.denoising_model = model

        # Initialize DeepGP model for GP regression
        self.deep_gp = DeepGPp(d_model, num_inducing)
        self.gp = gp
        self.sigma = nn.Parameter(torch.randn(1))

        # Layer normalization and feedforward networks
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Sequential(nn.Linear(d_model, d_model*4),
                                   nn.ReLU(),
                                   nn.Linear(d_model*4, d_model))
        self.ffn_2 = nn.Sequential(nn.Linear(d_model, d_model*4),
                                   nn.ReLU(),
                                   nn.Linear(d_model*4, d_model))

        self.d = d_model
        self.no_noise = no_noise
        self.iso = iso

    def add_gp_noise(self, x):
        """
        Add GP noise to the input using the DeepGP model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - x_noisy (Tensor): Noisy input with added GP noise.
        - dist (Tensor): GP distribution if GP is used.
        """
        b, s, _ = x.shape

        # Predict GP noise and apply layer normalization
        eps_gp, dist = self.deep_gp.predict(x)
        x_noisy = self.norm_1(x + self.ffn_1(eps_gp))

        return x_noisy, dist

    def forward(self, enc_inputs):
        """
        Forward pass of the BlurDenoiseModel.

        Args:
        - enc_inputs (Tensor): Encoder inputs.
        - dec_inputs (Tensor): Decoder inputs.

        Returns:
        - dec_output (Tensor): Denoised decoder output.
        - dist (Tensor): GP distribution if GP is used.
        """

        enc_noisy, dist_enc = self.add_gp_noise(enc_inputs)

        # Perform denoising with the underlying forecasting model
        enc_denoise = self.denoising_model(enc_noisy)

        # Apply layer normalization and feedforward network to the decoder output
        dec_output = self.norm_2(enc_inputs + self.ffn_2(enc_denoise))

        return dec_output, dist_enc


class ForecastBlurDenoise(nn.Module):
    def __init__(self, *, forecasting_model: nn.Module,
                 gp: bool = True,
                 iso: bool = False,
                 no_noise: bool = False,
                 add_noise_only_at_training: bool = False,
                 pred_len: int = 0,
                 num_inducing: int = 32,
                 d_model: int):
        """
        Forecast-blur-denoise Module.

        Args:
        - forecasting_model (nn.Module): The underlying forecasting model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - iso (bool): Flag indicating whether to use Gaussian isotropic for the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during denoising
         (denoise predictions directly).
        - add_noise_only_at_training (bool): Flag indicating whether to add noise only during training.
        - pred_len (int): Length of the prediction horizon.
        - src_input_size (int): Number of features in input.
        - tgt_output_size (int): Number of features in output.
        - num_inducing (int): Number of inducing points for GP model.
        - d_model (int): Dimensionality of the model (default is 32).
        """
        super(ForecastBlurDenoise, self).__init__()

        self.pred_len = pred_len
        self.add_noise_only_at_training = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.forecasting_model = forecasting_model

        # Initialize the blur and denoise model
        self.de_model = BlurDenoiseModel(self.forecasting_model,
                                         d_model,
                                         gp=gp,
                                         no_noise=no_noise,
                                         iso=iso,
                                         num_inducing=num_inducing)

        self.d_model = d_model

    def forward(self, enc_inputs):
        """
        Forward pass of the ForecastDenoising model.

        Args:
        - enc_inputs (Tensor): Encoder inputs.
        - dec_inputs (Tensor): Decoder inputs.
        - y_true (Tensor): True labels for training (optional).

        Returns:
        - final_outputs (Tensor): Model's final predictions.
        - loss (Tensor): Combined loss from denoising and forecasting components.
        """
        y_true = enc_inputs
        mll_error = 0
        loss = 0

        # Get outputs from the forecasting model
        enc_outputs = self.forecasting_model(enc_inputs)

        de_model_outputs, dist = self.de_model(enc_outputs.clone())

        # If using GP and during training, compute MLL loss
        if self.gp and self.training:

            mll = DeepApproximateMLL(
                VariationalELBO(self.de_model.deep_gp.likelihood, self.de_model.deep_gp, self.d_model))

            mll_error = -mll(dist, y_true).mean()

            loss = torch.clamp(mll_error, max=1.0, min=0.0)

        return de_model_outputs, loss

