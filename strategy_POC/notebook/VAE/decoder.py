import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *



def decoder(self, z):

    # is_training = get_current_tower_context().is_training
    # fc_l1 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.T * self.hps.f[1])

    out_l1 = fully_connected(inputs=z, num_outputs=self.hps.D * self.hps.f[0])
    rs_l1 = tf.reshape(out_l1, shape=[-1, self.hps.D, self.hps.f[0]])

    fc_l2 = fully_connected(inputs=rs_l1, num_outputs=self.hps.T)

    # [M, D, T] => [M, T, D]
    x_recon = tf.transpose(fc_l2, perm=[0, 2, 1])

    return x_recon

    

    # conv_l3 = conv1d(name='decode_l3', inputs=rs_l1, kernel_size=3, in_C=self.hps.f[0], out_C=self.hps.D)

    # deconv_l1 =  deconv2d(name='decode_l1', x, num_filters, filter_size=(3, 3), stride=(2, 2), pad="SAME", init_scale=0.1, init=False, mask=None, dtype=tf.float32, **_):
    
    # a_l1 = inverse_leaky_relu(rs_l1)

    # deconv_l1 = inverse_conv1d(name='l1', M=self.hps.M, T=self.hps.T, k=3, stride=1,
    #                    in_C=self.hps.f[0], out_C=self.hps.D, value=a_l1)

    # batch_l3 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l3)

    # x_recon = fully_connected(inputs=rs_l1, num_outputs=self.hps.D)
    # return deconv_l1