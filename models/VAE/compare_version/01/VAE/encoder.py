import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *


def encoder(self, x):
    is_training = get_current_tower_context().is_training

    # [M, T, D] => [M, T, f0]

    out_l1 = fully_connected(inputs=x, num_outputs=self.hps.f[0])

    rs_l1 = tf.reshape(out_l1, shape=[-1, self.hps.T * self.hps.f[0]])

    z_lst = fully_connected(inputs=rs_l1, num_outputs=2 * self.hps.n_z)


    z_mu, z_std1 = split(z_lst, split_dim=1, split_sizes=[
                         self.hps.n_z, self.hps.n_z])
    z_std = 1e-10 + tf.nn.softplus(z_std1)

    if self.hps.is_VAE:
        noise = tf.random_normal(shape=tf.shape(z_mu), mean=0.0, stddev=1.0)
        z = z_mu + noise * z_std
    else:
        z = z_mu

    return z_mu, z_std, z, z
