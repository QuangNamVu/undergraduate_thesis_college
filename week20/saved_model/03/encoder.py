import tensorflow as tf

num_neurons = 100


class Encoder(object):

    def __init__(self, dim_output):
        self.dim_output = dim_output

    def output(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            # # reshape first

            # x_reshape = tf.reshape(inputs, shape=[-1, 128, 8])
            #
            # conv1 = tf.layers.conv1d(inputs=x_reshape, filters=128, kernel_size=5, padding='same',
            #                          data_format='channels_last', activation='elu')  # N, T, 128
            #
            # batch1 = tf.layers.batch_normalization(momentum=0.5, inputs=conv1, axis=-1)  # N, T, 64
            #
            # elu1 = tf.nn.elu(features=batch1)
            #
            # flat1 = tf.reshape(elu1, shape=[-1, 128 * 16])

            flat1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=50, activation_fn=tf.nn.relu)

            z_mu = tf.contrib.layers.fully_connected(inputs=flat1, num_outputs=50, activation_fn=tf.nn.relu)

            z_lsgms = tf.contrib.layers.fully_connected(inputs=flat1, num_outputs=50, activation_fn=tf.nn.relu)

            return (z_mu, z_lsgms)
