import tensorflow as tf

from info_params import params

num_neurons = 100
info_params = params()


class Encoder(object):
    def __init__(self, params):
        self.params = params

    def output(self, inputs, reuse=False):
        N, T, D, n_z = self.params.n_batch, self.params.n_time_steps, self.params.d_inputs, self.params.n_z

        with tf.variable_scope('encoder', reuse=reuse):
            # # reshape first
            #
            # x_reshape = tf.reshape(inputs, shape=[-1, 128, 8])
            #
            # conv1 = tf.layers.conv1d(inputs=x_reshape, filters=64, kernel_size=5, padding='same',
            #                          data_format='channels_last', activation='elu')  # N, T, 64
            #
            # batch1 = tf.layers.batch_normalization(momentum=0.5, inputs=conv1, axis=-1)  # N, T, 64
            #
            # elu1 = tf.nn.elu(features=batch1)
            #

            flat_l1 = tf.reshape(inputs, shape=[-1, T * D])

            z_mu = tf.contrib.layers.fully_connected(inputs=flat_l1, num_outputs=n_z, activation_fn=tf.nn.relu)

            z_lsgms = tf.contrib.layers.fully_connected(inputs=flat_l1, num_outputs=n_z, activation_fn=tf.nn.relu)

            return (z_mu, z_lsgms)
