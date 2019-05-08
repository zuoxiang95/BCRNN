#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zuoxiang@jd.com
@file: train_attribute.py
@time: 2019/4/18 11:32
@desc:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim

from AFG_Net import AFGNet
from hparams import hparams as hp
from loss import attribute_weight
from loss import configure_learning_rate
from loss import configure_optimizer
from loss import category_classification_loss
from loss import attribute_classification_loss, sigmoid_cross_entropy_balanced
from build_tf_records import input_fn
from keras.applications.nasnet import nasnet
nasnet.NASNetLarge()

logging.basicConfig(level=10, filename='train.log')
logger = logging.getLogger(__name__)


def train(train_list, stage='category'):
    afg_net = AFGNet()

    images, category, attribute = input_fn(True, train_list, stage='classification', params={
        'num_epochs': hp.num_epochs,
        'num_category_classes': hp.num_category_classes,
        'num_attribute_classes': hp.num_attribute_classes,
        'batch_size': hp.batch_size,
        'buffer_size': hp.buffer_size,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': hp.image_size,
        'width': hp.image_size,
    })

    # build network
    images_input = tf.placeholder(tf.float32, shape=(None, hp.image_size, hp.image_size, 3))
    category_input = tf.placeholder(tf.int64, shape=(None, hp.num_category_classes))
    attribute_input = tf.placeholder(tf.int64, shape=(None, hp.num_attribute_classes))
    category_output, attribute_output = afg_net.buildNet(images_input, hp.num_category_classes,
                                                         hp.num_attribute_classes,
                                                         weight_decay=hp.weight_decay, is_training=True,
                                                         dropout_keep_prob=hp.dropout_keep_prob, stage=stage)

    # set optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = configure_learning_rate(hp.train_images_num, global_step)
    optimizer = configure_optimizer(learning_rate)

    # loss definition
    # attribute = tf.cast(attribute, tf.float32)
    # attribute_output = tf.cast(attribute_output, tf.float32)
    attribute_loss = sigmoid_cross_entropy_balanced(attribute_output, attribute)
    # attribute_loss = attribute_classification_loss(attribute_output, attribute, 5)
    category_loss, category_accuracy = category_classification_loss(category_output, category)
    total_loss = attribute_loss + category_loss
    # total_loss = attribute_loss
    slim.losses.add_loss(total_loss)
    loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    # trainable variables definition
    # do not train BCRNN
    # exclude = ['BCRNN', 'LandmarkAttention']
    exclude = []
    variables_to_train = [v for v in tf.trainable_variables()
                          if v.name.split('/')[0] not in exclude]

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainOp = optimizer.minimize(loss, global_step=global_step, var_list=variables_to_train)

    merge_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        logger.info('Training...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=hp.max_checkpoints)
        # ckpt = tf.train.get_checkpoint_state(hp.model_dir)

        exclude = ['ClothingAttention', 'Classification', 'global_step']
        logger.debug('variables to ignore:{}'.format(exclude))
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        init = slim.assign_from_checkpoint_fn(hp.model_dir + '/model.ckpt-4500', variables_to_restore,
                                              ignore_missing_vars=True)
        init(sess)

        # Initialization model
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     logger.info("Model restored...")
        # else:
        #     logger.info('Initialization...')
        #     sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(hp.model_dir, sess.graph)
        for i in range(hp.train_images_num):
            category_res, attribute_res = sess.run([category, attribute])
            _, train_summary, itr, train_loss = sess.run([trainOp, merge_summary, global_step, loss],
                                                         feed_dict={images_input: images.eval(),
                                                                    category_input: category_res,
                                                                    attribute_input: attribute_res})
            logger.info("itr: %d, loss: %f" % (itr, train_loss))
            train_writer.add_summary(train_summary, itr)

            if i % hp.max_checkpoints == 0:
                saver.save(sess, hp.model_dir + '/model.ckpt', itr)


if __name__ == '__main__':
    train(['/home/zuoxiang/AFG_Network/cloth.tfrecords'])
