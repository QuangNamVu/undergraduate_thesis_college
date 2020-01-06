#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ref: DCGAN.py Author: Yuxin Wu
# Author: Nam VQ

N_train_img = 1024
import argparse
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator

# from VAE.Model import VDEModelDesc, VDETrainer  # , RandomZData
from VAE.Model import ModelDesc, Trainer  # , RandomZData
from VAE.info_params import get_default_hparams
from VAE.load_data import *
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


class RandomEpsilon(DataFlow):
    def __init__(self, n_z):
        super(RandomEpsilon, self).__init__()
        self.n_z = n_z

    def __iter__(self):
        while True:
            yield [np.random.normal(loc=0, scale=1, size=[self.n_z, self.n_z])]


# prefetch data
def get_mnist_data():
    global hps

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data[0:N_train_img] / np.float32(255)
    # [N, -1] -> [N, C]
    train_labels = np.eye(hps.C)[np.reshape(train_labels, -1)]
    train_labels = train_labels.astype(np.int32)  # not required
    # [N, C] -> [N, 1, C]
    train_labels = np.reshape(train_labels, [-1, 1, hps.C])

    eval_data = eval_data / np.float32(255)
    eval_labels = np.eye(hps.C)[np.reshape(eval_labels, -1)]
    # [N, -1] -> [N, C]
    eval_labels = eval_labels.astype(np.int32)  # not required
    # [N, C] -> [N, 1, C]
    eval_labels = np.reshape(eval_labels, [-1, 1, hps.C])

    train_data = LoadData(train_data, train_data, train_labels, shuffle=True)
    test_data = LoadData(eval_data, eval_data, eval_labels, shuffle=False)

    ds_train = ConcatData([train_data])
    ds_test = ConcatData([test_data])

    # ds = train_data
    ds_train = BatchData(ds_train, batch_size=hps.M)
    ds_test = BatchData(ds_test, batch_size=hps.M)

    return ds_train, ds_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()

    # info_params = params()
    # M = VDEModelDesc(info_params)
    hps = get_default_hparams()
    hps.C = 10
    hps.T = 28
    hps.D = 28
    hps.n_z = 28
    M = ModelDesc(hps)

    logger.auto_set_dir(action='d')
    ds_train, ds_test = get_mnist_data()
    # sess = SessionCreatorAdapter(NewSessionCreator(), lambda sess: tf_debug.LocalCLIDebugWrapperSession(sess))
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "nam-pc:7000")

    creator = SessionCreatorAdapter(NewSessionCreator(),
                                    lambda sess: tf_debug.TensorBoardDebugWrapperSession(sess, "nam-pc:7000"))
    # Trainer(input=QueueInput(ds_train), model=M).train_with_defaults(
    #     callbacks=[
    #         ModelSaver(),
    #         callbacks.MergeAllSummaries(),
    #         MinSaver('total_loss'),
    #         InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
    #     ],
    #     steps_per_epoch=info_params.steps_per_epoch,
    #     max_epoch=info_params.epochs,
    #     # session_init=SaverRestore(args.load) if args.load else None

    # )

    Trainer(
        input=QueueInput(ds_train), model=M).train_with_defaults(
        callbacks=[
            ModelSaver(),
            callbacks.MergeAllSummaries(),
            MinSaver('total_loss'),
            # InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
            InferenceRunner(ds_test, [
                ScalarStats('predict_trend/accuracy_'),
                BinaryClassificationStats(pred_tensor_name='predict_trend/y_pred_one_hot',
                                          label_tensor_name='y_one_hot')])
        ],
        steps_per_epoch=hps.steps_per_epoch,
        max_epoch=hps.epochs,
        session_init=SaverRestore(args.load) if args.load else None
    )
