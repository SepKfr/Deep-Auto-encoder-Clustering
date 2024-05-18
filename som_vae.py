import torch
from torch import nn

from util_scores import get_scores


class LinearEncoder(nn.Module):
    def __init__(self, input_size=2, seq_len=30, hidden_size=100, enc_out_dim=30, *args, **kwargs):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(seq_len*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, enc_out_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_in = x.reshape(x.shape[0], -1)
        out = self.relu(self.fc1(x_in))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        # output shape should be: (batch_size, seq_len * num_sensors)
        return out


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=3, hidden_size=100, seq_len=30,
                 input_size=2, *args, **kwargs):
        super(LinearDecoder, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_len*input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # we want the shape (batch_size, seq_len, num_sensors)
        return out.reshape(-1, self.seq_len, self.input_size)


class SOMVAE(nn.Module):
    """SOM-VAE
    Model by Fortuin et al. (2019): https://arxiv.org/abs/1806.02199
    The official code in tensorflow: https://github.com/ratschlab/SOM-VAE
    Args:
        d_input: input dimension or sequence length
        d_channel: number of channels or sensors
        d_enc_dec: hidden dimension in encoder and decoder
        d_latent: dimension of encodings and embeddings
        d_som: dimension of the SOM as a list of two integers
        alpha: factor of the commitment loss
        beta: factor of the SOM loss
    """
    def __init__(self,
                 *,
                 d_input: int = 100,
                 d_channel: int = 3,
                 d_enc_dec: int = 100,
                 d_latent: int = 64,
                 device: torch.device = torch.device("cpu"),
                 n_clusters: int,
                 alpha: float = 1,
                 beta: float = 1,
                 lr: float = 1e-3,
                 ):
        super().__init__()

        self.device = device
        self.d_som = [3, 3]
        self.d_latent = d_latent
        self.n_clusters = n_clusters

        self.encoder = LinearEncoder(input_size=d_channel, seq_len=d_input,
                                     hidden_size=d_enc_dec, enc_out_dim=d_latent)
        self.decoder_e = LinearDecoder(latent_dim=d_latent, hidden_size=d_enc_dec,
                                       seq_len=d_input, input_size=d_channel)
        self.decoder_q = LinearDecoder(latent_dim=d_latent, hidden_size=d_enc_dec,
                                       seq_len=d_input, input_size=d_channel)

        self.embeddings = nn.Parameter(nn.init.trunc_normal_(torch.empty((self.d_som[0],
                                                                          self.d_som[1], d_latent)),
                                                             std=0.05, a=-0.1, b=0.1))
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):

        # encoding
        b, s_l = x.shape[0], x.shape[1]

        if len(x.shape) > 3:
            x = x.reshape(b, s_l, -1)

        z_e = self.encoder(x)
        # embedding
        z_q, z_dist, k = self._find_closest_embedding(z_e, batch_size=x.shape[0])
        z_q_neighbors = self._find_neighbors(z_q, k, batch_size=x.shape[0])

        x_q = self.decoder_q(z_q)
        x_e = self.decoder_e(z_e)

        loss = self.loss(x, x_e, x_q, z_e, z_q, z_q_neighbors)

        y = y[:, 0, :].reshape(-1)

        adj_rand_index, nmi, f1, p_score = get_scores(y, k, self.n_clusters, device=self.device)

        return loss, adj_rand_index, nmi, f1, p_score, x_q + x_e

    def _get_coordinates_from_idx(self, k):
        k_1 = torch.div(k, self.d_som[1], rounding_mode='floor')
        k_2 = k % self.d_som[1]
        return k_1, k_2

    def _find_neighbors(self, z_q, k, batch_size):
        k_1, k_2 = self._get_coordinates_from_idx(k)

        k1_not_top = k_1 < self.d_som[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.d_som[1] - 1
        k2_not_left = k_2 > 0

        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)

        z_q_up = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]

        z_q_down = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]

        z_q_right = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_right_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] = z_q_right_[k2_not_right == 1]

        z_q_left = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]

        return torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)

    @staticmethod
    def _gather_nd(params, idx):
        """Similar to tf.gather_nd. Here: returns batch of params given the indices."""
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def _find_closest_embedding(self, z_e, batch_size):
        """Picks the closest embedding for every encoding."""
        z_dist = (z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2
        z_dist_sum = torch.sum(z_dist, dim=-1)
        z_dist_flat = z_dist_sum.view(batch_size, -1)
        k = torch.argmin(z_dist_flat, dim=-1)
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_batch = torch.stack([k_1, k_2], dim=1)
        return self._gather_nd(self.embeddings, k_batch), z_dist_flat, k

    def _loss_reconstruct(self, x, x_e, x_q):
        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q
        return mse_l

    def _loss_commit(self, z_e, z_q):
        commit_l = self.mse_loss(z_e, z_q)
        return commit_l

    @staticmethod
    def _loss_som(z_e, z_q_neighbors):
        z_e = z_e.detach()
        som_l = torch.mean((z_e.unsqueeze(1) - z_q_neighbors) ** 2)
        return som_l

    def loss(self, x, x_e, x_q, z_e, z_q, z_q_neighbors):
        mse_l = self._loss_reconstruct(x, x_e, x_q)
        commit_l = self._loss_commit(z_e, z_q)
        som_l = self._loss_som(z_e, z_q_neighbors)
        # loss = mse_l + self.hparams.alpha * commit_l + self.hparams.beta * som_l
        raw_loss = mse_l + commit_l + som_l
        return raw_loss