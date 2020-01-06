import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *


def encoder(self, x):
    is_training = get_current_tower_context().is_training

    # [M, T, D] => [M, T, 32]

    out_l1 = conv1d(name='encode_l1', inputs=x, kernel_size=5, stride=1,
                    in_C=self.hps.D, out_C=32)

    # [M, T, D] => [M, 30, 64]
    out_l2 = conv1d(name='encode_l2', inputs=out_l1, kernel_size=3, stride=2,
                    in_C=32, out_C=64)
    
    # [M, 15, 64] => [M, 24, 2*n_z]
    out_l3 = conv1d(name='encode_l3', inputs=out_l2, kernel_size=7, stride=1,
                    in_C=64, out_C= 2 * self.hps.n_z, padding="VALID")

    rs_l1 = tf.reshape(out_l3, shape=[-1, 24 * 2 * self.hps.n_z])

    z_lst = fully_connected(inputs=rs_l1, num_outputs=2 * self.hps.n_z)

    z_mu, z_std1 = split(z_lst, split_dim=1, split_sizes=[
                         self.hps.n_z, self.hps.n_z])
    z_std = 1e-10 + tf.nn.softplus(z_std1)

    if self.hps.is_VAE:
        noise = tf.random_normal(shape=tf.shape(z_mu), mean=0.0, stddev=1.0)
        z = z_mu + noise * z_std
    else:
        z = z_mu

    return z_mu, z_std, z, z_mu
