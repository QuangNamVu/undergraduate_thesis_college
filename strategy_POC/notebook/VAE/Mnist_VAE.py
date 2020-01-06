#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ref: DCGAN.py Author: Yuxin Wu
# Author: Nam VQ


import argparse
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator

from Model import VDEModelDesc, VDETrainer  # , RandomZData
from info_params import get_default_hparams
from load_data import *
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


# prefetch data
def get_mnist_data(hps):
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data[0:hps.N_train_seq] / np.float32(255)
    train_labels = train_labels[0:hps.N_train_seq]
    train_labels = np.eye(hps.C)[np.reshape(train_labels, -1)].astype(np.int32)
    train_labels = np.expand_dims(train_labels, axis=1)

    eval_data = eval_data / np.float32(255)
    eval_labels = np.eye(hps.C)[np.reshape(eval_labels, -1)].astype(np.int32)
    eval_labels = np.expand_dims(eval_labels, axis=1)

    train_data = LoadData(train_data, train_data, train_labels, shuffle=True)
    test_data = LoadData(eval_data, eval_data, eval_labels, shuffle=False)

    ds_train = ConcatData([train_data])
    ds_test = ConcatData([test_data])

    # ds = train_data
    ds_train = BatchData(ds_train, batch_size=hps.M)
    ds_test = BatchData(ds_test, batch_size=hps.M)

    return ds_train, ds_test


if __name__ == '__main__':
    hps = get_default_hparams()
    hps.C = 10
    hps.D = 28
    hps.T = 28
    hps.N_train_seq = 100
    # hps.M = 100
    # hps.n_z = 10
    # hps.Tau = 1
    # hps.is_VAE = True
    # hps.is_VDE = False
    # hps.steps_per_epoch = 100


    M = VDEModelDesc(hps)

    logger.auto_set_dir(action='d')
    ds_train, ds_test = get_mnist_data(hps)
    # sess = SessionCreatorAdapter(NewSessionCreator(), lambda sess: tf_debug.LocalCLIDebugWrapperSession(sess))
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "nam-pc:7000")

    creator = SessionCreatorAdapter(NewSessionCreator(),
                                    lambda sess: tf_debug.TensorBoardDebugWrapperSession(sess, "nam-pc:7000"))
    VDETrainer(input=QueueInput(ds_train), model=M).train_with_defaults(
        callbacks=[
            ModelSaver(),
            callbacks.MergeAllSummaries(),
            MinSaver('total_loss'),
            InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
        ],
        steps_per_epoch=hps.steps_per_epoch,
        max_epoch=hps.epochs,
    )
