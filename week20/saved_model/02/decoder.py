import tensorflow as tf


# p(x_hat | z)

class Decoder(object):
    def __init__(self, dim_output):
        self.dim_output = dim_output

    def output(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            z_reshape = tf.expand_dims(inputs, 1)  # N, 1, 50

            lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=12, activation=tf.nn.relu),
                output_size=1024)  # N, 1, 1024

            x_hat, _ = tf.nn.dynamic_rnn(lstm_cell, z_reshape, dtype=tf.float32)

            x_hat_flat = tf.reshape(x_hat, shape=[-1, 1024])

            # x_hat_flat = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=1024, activation_fn=tf.nn.relu)

            return x_hat_flat
