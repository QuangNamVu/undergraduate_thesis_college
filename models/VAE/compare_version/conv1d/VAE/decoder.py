import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *



def decoder(self, z):

    # is_training = get_current_tower_context().is_training
    # fc_l1 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.T * self.hps.f[1])
    rs_l3 = gaussian_dense(name='decode_fc1', inputs=z, out_C= 24 * 2 * self.hps.n_z)
    next_seq = tf.reshape(rs_l3, shape=[-1, 24, 2 * self.hps.n_z])


    out_l3 = inverse_conv1d(name='decode_l3', M=self.hps.M, T=30, k=7, stride=1,
                            in_C=2 *self.hps.n_z, out_C=64, value=next_seq, padding = 'VALID')
    
    out_l2 = inverse_conv1d(name='decode_l2', M=self.hps.M, T=60, k=3, stride=2,
                            in_C=64, out_C=32, value=out_l3)

    out_l1 = inverse_conv1d(name='decode_l1', M=self.hps.M, T=60, k=3, stride=1,
                            in_C=32, out_C=self.hps.D, value=out_l2)

    # out_l1 = fully_connected(inputs=z, num_outputs=self.hps.T * self.hps.f[0])
    # rs_l1 = tf.reshape(out_l1, shape=[-1, self.hps.T, self.hps.f[0]])
    # x_recon = fully_connected(inputs=rs_l1, num_outputs=self.hps.D)

    return out_l1
