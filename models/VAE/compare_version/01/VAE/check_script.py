from __future__ import print_function

import tensorflow as tf
import numpy as np

M = 64
T = 120
D = 3
lstm_units = 10
n_z = 20

X = np.random.randn(M, T, D).astype(np.float32)

X_tf = tf.constant(X, dtype=tf.float32)
cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)

outputs, states = tf.nn.dynamic_rnn(
    cell=cell,
    inputs=X_tf,
    sequence_length=[T] * M,
    dtype=tf.float32
)

z_lst = tf.contrib.layers.fully_connected(inputs=states.c, num_outputs=2 * n_z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    outputs_val = sess.run(outputs)
    states_val = sess.run(states)
    print(outputs_val)
