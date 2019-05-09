#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: loss.py
@time: 2019/4/18 14:33
@desc:
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from hparams import hparams as hp

attribute_weight = [10.0 for i in range(1000)]


def category_classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy


def sigmoid_cross_entropy_balanced(logits, label, name='cross_entrony_loss'):
    """
    Initially proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1.-y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost)

    return cost


def attribute_classification_loss(logit, label, weight, name=None):
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(label, logit, weight))
    return loss


def attribute_classification_loss_1(logit, label, weight, name=None):
    with ops.name_scope(name, "logistic_loss", [logit, label, weight]) as name:
        logit = ops.convert_to_tensor(logit, name="logit")
        label = ops.convert_to_tensor(label, name="label")
        weight = ops.convert_to_tensor(weight, name="weight")
        try:
            label.get_shape().merge_with(logit.get_shape())
        except ValueError:
            raise ValueError(
                "logits and targets must have the same shape (%s vs %s)" %
                (logit.get_shape(), label.get_shape()))
    # loss = math_ops.add((1 - weight) * (1 - label) * logit, (1 - label - weight + 2 * weight * label) * (
    #     math_ops.log1p(math_ops.exp(-math_ops.abs(logit)))))
    log_weight = 1.0 + (weight - 1.0) * label
    loss = math_ops.add(
        (1.0 - label) * logit,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logit))) +
                      nn_ops.relu(-logit)),
        name=name)

    return tf.reduce_mean(loss)


def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / hp.batch_size *
                      hp.num_epochs_per_decay)
    if hp.sync_replicas:
        decay_steps /= hp.replicas_to_aggregate

    if hp.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(hp.learning_rate,
                                          global_step,
                                          decay_steps,
                                          hp.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif hp.learning_rate_decay_type == 'fixed':
        return tf.constant(hp.learning_rate, name='fixed_learning_rate')
    elif hp.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(hp.learning_rate,
                                         global_step,
                                         decay_steps,
                                         hp.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         hp.learning_rate_decay_type)


def configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if hp.optimizer is not recognized.
    """
    if hp.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=hp.adadelta_rho,
            epsilon=hp.opt_epsilon)
    elif hp.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=hp.adagrad_initial_accumulator_value)
    elif hp.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=hp.adam_beta1,
            beta2=hp.adam_beta2,
            epsilon=hp.opt_epsilon)
    elif hp.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=hp.ftrl_learning_rate_power,
            initial_accumulator_value=hp.ftrl_initial_accumulator_value,
            l1_regularization_strength=hp.ftrl_l1,
            l2_regularization_strength=hp.ftrl_l2)
    elif hp.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=hp.momentum,
            name='Momentum')
    elif hp.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=hp.rmsprop_decay,
            momentum=hp.rmsprop_momentum,
            epsilon=hp.opt_epsilon)
    elif hp.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', hp.optimizer)
    return optimizer
