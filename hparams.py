#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: hparams.py
@time: 2019/4/16 17:54
@desc:
"""

import tensorflow as tf

hparams = tf.contrib.training.HParams(
    # Training:
    image_size=224,
    num_category_classes=48,
    num_attribute_classes=1000,
    num_epochs=20,
    batch_size=10,
    buffer_size=100,    # The number of input images in buffer.
    train_images_num=207300,
    dropout_keep_prob=0.5,
    model_dir='model',
    init=False,
    train_stage='landmark',
    max_checkpoints=1000,

    # Optimization Flags
    weight_decay=0.0005,
    # The name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop"
    optimizer='adadelta',
    # adadelta parameters
    adadelta_rho=0.95,
    # adagrad parameters
    adagrad_initial_accumulator_value=0.1,
    # adam parameters
    adam_beta1=0.9,
    adam_beta2=0.999,
    opt_epsilon=1.0,
    # ftrl parameters
    ftrl_learning_rate_power=-0.5,
    ftrl_initial_accumulator_value=0.1,
    ftrl_l1=0.0,
    ftrl_l2=0.0,
    # momentum parameters
    momentum=0.9,
    # rmsprop parameters
    rmsprop_momentum=0.9,
    rmsprop_decay=0.9,

    # Learning Rate Flags
    # Specifies how the learning rate is decayed. One of "fixed", "exponential", or "polynomial"
    learning_rate_decay_type='exponential',
    learning_rate=0.001,
    end_learning_rate=0.00001,
    label_smoothing=0.0,
    learning_rate_decay_factor=0.94,
    num_epochs_per_decay=2.0,
    sync_replicas=False,
    replicas_to_aggregate=1,
    moving_average_decay=None,   # The decay to use for the moving average.
    image_path='/home/zuoxiang/data/deepfashion/Img'
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
