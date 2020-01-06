import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
import math


def get_noise(shape=None):
    return tf.random_normal(shape=shape, mean=0.0, stddev=1.0)


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_mean_std(x, axis=None, keepdims=False):
    mean = tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    std = tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

    return mean, std


def auto_corr_loss(z, z_tau, n_dim=3):
    if n_dim is 3:
        z_norm = tf.norm(z, ord='euclidean', axis=[1, 2], keepdims=True)
        z_tau_norm = tf.norm(z_tau, ord='euclidean', axis=[1, 2], keepdims=True)
        z_mean = tf.reduce_mean(z, axis=[1, 2], keepdims=True)
        z_tau_mean = tf.reduce_mean(z_tau, axis=[1, 2], keepdims=True)
        num = (z - z_mean) * (z_tau - z_tau_mean)
        den = z_norm * z_tau_norm

    elif n_dim is 2:
        z_mean, z_std = reduce_mean_std(z, axis=[1], keepdims=True)
        z_tau_mean, z_tau_std = reduce_mean_std(z_tau, axis=[1], keepdims=True)
        num = (z - z_mean) * (z_tau - z_tau_mean)
        den = z_std * z_std

    return tf.reduce_mean(- num / den, name='auto_corr')


def inverse_elu(x):
    return tf.where(x >= 0.0, x, .5 * tf.log(tf.square(x + 1)))


def inverse_leaky_relu(x, alpha=0.2):
    if alpha <= 0.0:
        return tf.where(x >= 0.0, x, 0.0)

    return tf.where(x >= 0.0, x, x / alpha)


def inverse_tanh(x):
    return .25 * tf.log((1 + x) * (1 + x) / ((1 - x) * (1 - x)))


def inverse_conv1d(name, M, T, k, in_C, out_C, value, stride=1, padding='SAME'):
    filter = tf.get_variable(name, [k, out_C, in_C],
                             initializer=tf.keras.initializers.lecun_uniform(seed=None)
                             # initializer=tf.random_normal_initializer(0, 0.05, dtype=tf.float32)
                             )  # [kernel_width, output_depth, input_depth]
    conv_layer = tf.contrib.nn.conv1d_transpose(filter=filter, value=value,
                                                output_shape=[M, T, out_C],
                                                stride=stride, padding=padding)

    return conv_layer


def gaussian_dense(name, inputs, out_C):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)
    w = tf.get_variable('w_' + name, [inputs.get_shape()[-1], out_C], initializer=w_init)
    b = tf.get_variable('b_' + name, [out_C], initializer=b_init)
    h = tf.tensordot(inputs, w, axes=1) + b
    return h


def correlation_coefficient(y_true, y_pred):
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r' in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
        return 1 - pearson_r ** 2


def maxpool_l1(input, input_name, pool_size=2, strides=2):
    input = tf.identity(input, name=input_name)
    pool = tf.layers.max_pooling1d(inputs=input, pool_size=pool_size, strides=strides,
                                   data_format='channels_last')

    return pool


@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradGradWithArgmax(op, grad):
    print(len(op.outputs))
    print(len(op.inputs))
    print(op.name)
    return (array_ops.zeros(
        shape=array_ops.shape(op.inputs[0]),
        dtype=op.inputs[0].dtype), array_ops.zeros(
        shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
            gen_nn_ops.max_pool_grad_grad_with_argmax(
                op.inputs[0],
                grad,
                op.inputs[2],
                op.get_attr("ksize"),
                op.get_attr("strides"),
                padding=op.get_attr("padding")))


def unpool1d(origin_name, pool_value, k=2, stride=2):
    """
    :param origin_name: A string point to 3D tensor with shape [M, T, D]
                      contain argmax indices
    :param pool_value: A 3D tensor with shape [M, T//stride, D]
    :return: unpooling_value: A 3D tensor with shape [M, T, D]
    """
    origin_name += ":0"
    mask_value = tf.get_default_graph().get_tensor_by_name(origin_name)
    mask_value = tf.expand_dims(mask_value, axis=2)
    pool_value = tf.expand_dims(pool_value, axis=2)
    k_sizes = [1, k, 1, 1]
    strides = [1, stride, 1, 1]
    unpool = gen_nn_ops.max_pool_grad(mask_value, pool_value, pool_value, k_sizes, strides, 'VALID')
    unpool = tf.squeeze(unpool, axis=2)
    return unpool


def batch_norm(x, hps, origin_name='bn'):
    # input_sh = x.get_shape().as_list()
    # input_sh[0] = hps.M
    input_sh = [hps.M, 1, 1]
    scale = tf.get_variable(origin_name + "_Y", input_sh, tf.float32,
                            tf.random_normal_initializer(0, 0.05, dtype=tf.float32))
    b = tf.get_variable(origin_name + "_b", input_sh, tf.float32,
                        tf.random_normal_initializer(0, 0.05, dtype=tf.float32))

    mean, var = tf.nn.moments(x, [0], keep_dims=True)
    mean = tf.identity(mean, name=origin_name + "_mean")
    var = tf.identity(var, name=origin_name + "_var")

    y = tf.nn.batch_normalization(x, mean=mean, variance=var, offset=b, scale=scale, variance_epsilon=1e-10)
    return y


def inverse_batch_norm(y, hps, origin_name='encode/bn'):
    scale = tf.get_default_graph().get_tensor_by_name(origin_name + "_Y:0")
    b = tf.get_default_graph().get_tensor_by_name(origin_name + "_b:0")

    running_mean = tf.get_default_graph().get_tensor_by_name(origin_name + "_mean:0")
    running_var = tf.get_default_graph().get_tensor_by_name(origin_name + "_var:0")

    # x_hat = tf.div_no_nan(y - b, scale)
    # x_hat = tf.where(tf.less(tf.square(scale), 1e-10), scale, (y-b) / scale)

    # x_hat = (y - b) / tf.clip_by_value(scale, 1e-5, 1 - 1e-5)
    x_hat = (y - b) / scale
    x = x_hat * running_var + running_mean

    return x


def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret
