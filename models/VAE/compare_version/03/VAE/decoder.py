import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *



def decoder(self, z):

    # is_training = get_current_tower_context().is_training
    # fc_l1 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.T * self.hps.f[1])

    out_l1 = fully_connected(inputs=z, num_outputs=self.hps.T * self.hps.f[0])
    rs_l1 = tf.reshape(out_l1, shape=[-1, self.hps.T, self.hps.f[0]])
    x_recon = fully_connected(inputs=rs_l1, num_outputs=self.hps.D)

    return x_recon
