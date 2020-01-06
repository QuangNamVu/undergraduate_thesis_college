import tensorflow as tf
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *


def predict(self):
    is_training = get_current_tower_context().is_training

    state_c = tf.layers.dropout(self.state_c, rate=self.hps.dropout_rate, name='pred_drop0', training=is_training)
    # [M, o] => [M, Tau * C]
    fc_l1 = tf.contrib.layers.fully_connected(inputs=state_c, num_outputs=self.hps.Tau * self.hps.C,
                                              activation_fn=tf.nn.elu)

    # [M, Tau * C] => [M, Tau, C]
    y_predict = tf.reshape(fc_l1, shape=[-1, self.hps.Tau, self.hps.C], name = 'y_pred')
    return y_predict
