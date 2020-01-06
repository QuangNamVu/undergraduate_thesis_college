import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *

d = 64


def encoder(self, x):
    # [_, T, D] => [_, T, d]
    out_l1 = conv1d(name='encode_l1', inputs=x, kernel_size=11, stride=1,
                    in_C=6, out_C=d)

    # [_, T, d] => [_, 15, d]
    out_l2 = conv1d(name='encode_l2', inputs=out_l1, kernel_size=5, stride=1,
                    in_C=d, out_C= d)

    # [_, 15, 2d] => [_, 8, 2d]
    out_l4 = conv1d(name='encode_l4', inputs=out_l2, kernel_size=5, stride=2,
                    in_C=d, out_C=2 * d)

    # [_, 8, 2d] => [_, 2, 4z]
    out_l5 = conv1d(name='encode_l5', inputs=out_l4, kernel_size=7, stride=1,
                    in_C=2 * d, out_C=4 * self.hps.n_z, padding="VALID")

    # [M, T, o] => [M, T * o] => [M, n_z]
    rs_l6 = tf.reshape(out_l5, shape=[-1, self.hps.lst_T[-1] * 4 * self.hps.n_z])
    z_lst = gaussian_dense(name='encode_fc1', inputs=rs_l6, out_C=2 * self.hps.n_z)
    z_mu, z_std1 = split(z_lst, split_dim=1, split_sizes=[self.hps.n_z, self.hps.n_z])
    # z_std = 1e-10 + tf.nn.softplus(z_std1)
    z_std = tf.nn.softplus(z_std1 + tf.log(tf.exp(1.0) - 1))

    if self.hps.is_VAE:
        # noise = tf.random_normal(shape=tf.shape(z_mu), mean=0.0, stddev=1.0)
        # z = z_mu + noise * z_std
        z = tf.contrib.distributions.MultivariateNormalDiag(loc=z_mu, scale_diag=z_std)
        z = z.sample()
    else:
        z = z_mu

    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hps.lstm_units, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, out_l1, sequence_length=[self.hps.lst_T[0]] * self.hps.M, dtype=tf.float32)

    return z_mu, z_std, z, state.c
