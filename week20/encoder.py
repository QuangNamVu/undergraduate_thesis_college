import tensorflow as tf

from info_params import params

info_params = params()


class Encoder(object):
    def __init__(self, params):
        self.params = params

    def output(self, inputs, reuse=False):
        M, T, D = inputs.shape
        n_z = self.params.n_z

        with tf.variable_scope('encoder', reuse=reuse):
            # reshape first

            conv_l1 = tf.layers.conv1d(inputs=inputs, filters=64, kernel_size=5, padding='same',
                                       data_format='channels_last', activation='elu')  # M, T, 64

            batch_l1 = tf.layers.batch_normalization(momentum=0.5, inputs=conv_l1, axis=-1)  # M, T, 64

            elu_l1 = tf.nn.elu(features=batch_l1)

            flat_l1 = tf.reshape(elu_l1, shape=[-1, T * 64])  # M, T*64

            z_mu = tf.contrib.layers.fully_connected(inputs=flat_l1, num_outputs=n_z, activation_fn=tf.nn.relu)

            z_lsgms = tf.contrib.layers.fully_connected(inputs=flat_l1, num_outputs=n_z, activation_fn=tf.nn.relu)

            return (z_mu, z_lsgms)
