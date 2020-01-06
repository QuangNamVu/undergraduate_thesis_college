import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *


def encoder(self, x):
    # if tf.dtype(x) == 'float64':
    is_training = get_current_tower_context().is_training

    # [M, T, D] => [M, D, T]
    rs_l1 = tf.transpose(x, perm=[0, 2, 1])

    # [M, D, T] => [M, D, f0]
    out_l1 = fully_connected(inputs=rs_l1, num_outputs=self.hps.f[0])

    # conv_l1 = conv1d(name='encode_l1', inputs=x, kernel_size=3, in_C=self.hps.D, out_C=self.hps.f[0])

    # conv_l1 =  tf.nn.conv1d(value=x, filters=[1, 1, 3], stride=1, padding="SAME", data_format='NWC')

    # conv_l1 = tf.nn.conv1d(value=x, filters=3, stride=1, padding="SAME")
    # conv_l1 = tf.nn.conv1d(value = x, filter = [self.hps.M, self.hps.T,self.hps.f[0]], stride=1, padding="SAME")

    # conv_l1 = tf.layers.Conv1D()

    # batch_l1 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l1)
    # out_l1 = tf.nn.leaky_relu(conv_l1)

    # z_lst = gaussian_dense(
    #     name='encode_fc2', inputs=out_l1, out_C=2 * self.hps.n_z)

    # [M, D, f0] => [M, D * f0] => [M, n_z]
    rs_l1 = tf.reshape(out_l1, shape=[-1, self.hps.D * self.hps.f[0]])
    z_lst = fully_connected(inputs=rs_l1, num_outputs=2 * self.hps.n_z)

    z_mu, z_std1 = split(z_lst, split_dim=1, split_sizes=[
                         self.hps.n_z, self.hps.n_z])
    z_std = 1e-10 + tf.nn.softplus(z_std1)

    if self.hps.is_VAE:
        noise = tf.random_normal(shape=tf.shape(z_mu), mean=0.0, stddev=1.0)
        z = z_mu + noise * z_std
    else:
        z = z_mu


    # return z_mu, z_std, z, state_c
    return z_mu, z_std, z, z

    # # z: [M, T, o]
    # # h: [M, o]
    # # c: [M, o]
    # # [M, T, f1] => [M, T, o]
    # cell = tf.nn.rnn_cell.LSTMCell(
    #     num_units=self.hps.lstm_units, state_is_tuple=True)

    # input_lstm = tf.reshape(z, shape=[-1, self.hps.T, 1])

    # _, state = tf.nn.dynamic_rnn(cell, input_lstm, sequence_length=[self.hps.T] * self.hps.M, dtype=tf.float32,
    #                              parallel_iterations=64)

    # state_c = state.c