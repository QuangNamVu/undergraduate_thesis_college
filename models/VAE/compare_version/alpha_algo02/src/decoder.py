import tensorflow as tf
from tensorpack import *
from tensorpack.models.pool import UnPooling2x2ZeroFilled
from tf_utils.ar_layers import *
from tf_utils.common import *

d = 64


def decoder(self, z):
    # [M, n_z] => [M, T, o]
    rs_l3 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.lst_T[-1] * 4 * self.hps.n_z)
    next_seq = tf.reshape(rs_l3, shape=[-1, self.hps.lst_T[-1], 4 * self.hps.n_z])

    # [_, 2, 4z] => [_, 8, 2d]
    out_l5 = inverse_conv1d(name='decode_l5', M=self.hps.M, T=self.hps.lst_T[-2], k=7, stride=1,
                            in_C=4 * self.hps.n_z, out_C=2 * d, value=next_seq, padding='VALID')

    # [_, 8, 2d] => [_, 15, 2d]
    out_l4 = inverse_conv1d(name='decode_l4', M=self.hps.M, T=self.hps.lst_T[-3], k=5, stride=2,
                            in_C=2 * d, out_C=d, value=out_l5)

    # [_, 15, 2d] => [_, 15, d]
    out_l3 = inverse_conv1d(name='decode_l3', M=self.hps.M, T=self.hps.lst_T[-4], k=5, stride=1,
                            in_C=d, out_C=d, value=out_l4)

    # [_, T, d] => [_, T, D]
    out_l1 = inverse_conv1d(name='decode_l1', M=self.hps.M, T=self.hps.T, k=5, stride=1,
                            in_C=d, out_C=6, value=out_l3)

    return out_l1
