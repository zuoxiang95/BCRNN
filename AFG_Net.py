#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: AFG_Net.py
@time: 2019/4/8 19:55
@desc:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn.python.ops.rnn_cell import _conv
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors, LSTMStateTuple
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util


class VGG(object):
    def __init__(self):
        self.default_image_size = 224

    def buildNet(self, netName, images, num_classes, dropout_keep_prob=0.5,
                 is_training=False, weight_decay=0.5, final_endpoint='conv5'):
        arg_scope = self.vgg_arg_scope(weight_decay=weight_decay)

        networks_map = {'VGG_11': self.vgg_a,
                        'VGG_16': self.vgg_16,
                        'VGG_19': self.vgg_19,
                        }
        with slim.arg_scope(arg_scope):
            func = networks_map[netName]
            logits, end_points = func(images, num_classes,
                                      dropout_keep_prob=dropout_keep_prob,
                                      is_training=is_training,
                                      final_endpoint=final_endpoint)
        return logits, end_points

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.
        Args:
          weight_decay: The l2 regularization coefficient.
        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def vgg_a(self, inputs,
              num_classes=1000,
              is_training=True,
              dropout_keep_prob=0.5,
              spatial_squeeze=True,
              scope='vgg_a',
              fc_conv_padding='VALID',
              global_pool=False,
              final_endpoint='conv5'):
        """Oxford Net VGG 11-Layers version A Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)
        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

    def vgg_16(self, inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False,
               final_endpoint='conv5'):
        """Oxford Net VGG 16-Layers version D Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)
        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
          end_points: a dict of tensors with intermediate activations.
        """

        def add_and_check_final(name, net):
            end_points[name] = net
            # print('{}_shape:{}'.format(name, net.shape))
            return name == final_endpoint

        end_points = {}
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                if add_and_check_final('conv1', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                if add_and_check_final('conv2', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                if add_and_check_final('conv3', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                if add_and_check_final('conv4', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                if add_and_check_final('conv5', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

    def vgg_19(self, inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_19',
               fc_conv_padding='VALID',
               global_pool=False,
               final_endpoint='conv5'):
        """Oxford Net VGG 19-Layers version E Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)
        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the non-dropped-out input to the logits layer (if num_classes is 0 or
            None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points


class CRNN(tf.contrib.rnn.RNNCell):
    def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape,
                 use_bias=True, initializers=None, name="crnn_cell"):
        """Construct CRNN.
        Args:
          conv_ndims: Convolution dimensionality (1, 2 or 3).
          input_shape: Shape of the input as int tuple, excluding the batch size, time steps and channel.
          output_channels: int, number of output channels of the conv.
          kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
          use_bias: (bool) Use bias in convolutions.
          skip_connection: If set to `True`, concatenate the input to the
            output of the conv LSTM. Default: `False`.
          forget_bias: Forget bias.
          initializers: Unused.
          name: Name of the module.
        Raises:
          ValueError: If `skip_connection` is `True` and stride is different from 1
            or if `input_shape` is incompatible with `conv_ndims`.
        """
        super(CRNN, self).__init__(name=name)

        if conv_ndims != len(input_shape) - 1:
            raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
                input_shape, conv_ndims))

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias

        self._state_size = input_shape
        self._output_size = input_shape

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        # return _zero_state_tensors(state_size, batch_size, dtype)

        # def expand(x, dim, N):
        #     return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)
        #
        with tf.variable_scope('CRNN_init', reuse=tf.AUTO_REUSE):
            state = tf.get_variable('zero_state',
                                    self.state_size,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))

        return tf.tile(tf.expand_dims(state, 0), [batch_size, 1, 1, 1])

    def __call__(self, inputs, state, scope=None):
        # r=array_ops.concat(axis=0, values=[inputs, state])
        new_hidden = _conv([inputs, state], self._kernel_shape,
                           self._output_channels, self._use_bias)
        # new_hidden = slim.conv2d(tf.concat([inputs, state], axis=0), 1, 3,
        #                          padding='SAME', scope='conv')
        output = math_ops.tanh(new_hidden)

        return output, output


class AFGNet(object):
    def __init__(self):
        self.default_image_size = 224
        self.vgg = VGG()

    def buildNet(self, images, category_classes, attribute_classes, weight_decay=0.0005,
                 is_training=False, dropout_keep_prob=0.5, stage='landmark'):

        # construct VGG base net
        net, end_points = self.vgg.buildNet('VGG_16', images, category_classes,
                                            is_training=is_training,
                                            weight_decay=weight_decay,
                                            dropout_keep_prob=dropout_keep_prob,
                                            final_endpoint='conv4')

        with tf.variable_scope('BCRNN'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                padding='SAME'):
                # 8 landmarks and 1 background
                heat_maps = slim.conv2d(net, 9, [1, 1], scope='ConstructHeatMaps')
                heat_maps = tf.sigmoid(heat_maps, name='sigmoid')

            # if stage.lower() == 'landmark':
            #     return heat_maps

            # heat-maps l-collar l-sleeve l-waistline l-hem r-...
            heat_maps = tf.transpose(heat_maps, (3, 0, 1, 2))
            # grammar:
            # RK:
            #         l.collar <-> l.waistline <-> l.hem;
            #         l.collar <-> l.sleeve;
            #         r.collar <-> r.waistline <-> r.hem;
            #         r.collar <-> r.sleeve:
            # RS:
            #         l.collar <-> r.collar;
            #         l.sleeve <-> r.sleeve;
            #         l.waistline <-> r.waistline;
            #         l.hem <-> r.hem:
            RK1_refined_heatmaps = self.BCRNNBlock(heat_maps, 3, [0, 2, 3], 'RK_1')
            RK2_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [0, 1], 'RK_2')
            RK3_refined_heatmaps = self.BCRNNBlock(heat_maps, 3, [4, 6, 7], 'RK_3')
            RK4_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [4, 5], 'RK_4')

            RS1_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [0, 4], 'RS_1')
            RS2_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [1, 5], 'RS_2')
            RS3_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [2, 6], 'RS_3')
            RS4_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [3, 7], 'RS_4')

            background = heat_maps[8]

            # max merge heatmaps
            l_collar = tf.reduce_max([RK1_refined_heatmaps[0], RK2_refined_heatmaps[0], RS1_refined_heatmaps[0]],
                                     axis=0)
            l_sleeve = tf.reduce_max([RK2_refined_heatmaps[1], RS2_refined_heatmaps[0]], axis=0)
            l_waistline = tf.reduce_max([RK1_refined_heatmaps[1], RS3_refined_heatmaps[0]], axis=0)
            l_hem = tf.reduce_max([RK1_refined_heatmaps[2], RS4_refined_heatmaps[0]], axis=0)

            r_collar = tf.reduce_max([RK3_refined_heatmaps[0], RK4_refined_heatmaps[0], RS1_refined_heatmaps[1]],
                                     axis=0)
            r_sleeve = tf.reduce_max([RK4_refined_heatmaps[1], RS2_refined_heatmaps[1]], axis=0)
            r_waistline = tf.reduce_max([RK3_refined_heatmaps[1], RS3_refined_heatmaps[1]], axis=0)
            r_hem = tf.reduce_max([RK3_refined_heatmaps[2], RS4_refined_heatmaps[1]], axis=0)

            refined_heatmaps = tf.stack([l_collar, l_sleeve, l_waistline, l_hem,
                                         r_collar, r_sleeve, r_waistline, r_hem,
                                         background], axis=3)

            # landmarks predictions
            output = tf.nn.softmax(refined_heatmaps, name='RefinedHeatMaps')

        if stage.lower() == 'landmark':
            return output, None

        with tf.variable_scope('LandmarkAttention'):
            output = output[:, :, :, :-1]
            AL = tf.reduce_mean(output, axis=-1, keep_dims=True)
            # tile_shape = tf.ones_like(output.shape)
            # tile_shape[-1] = output.shape[-1]
            AL = tf.tile(AL, [1, 1, 1, net.shape[-1]])
            GL = tf.multiply(AL, net)

        with tf.variable_scope('ClothingAttention'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer(),
                                scope='ClothingAttention'):
                AC = slim.max_pool2d(net, [2, 2], scope='AC_pool1')
                AC = slim.conv2d(AC, 512, [3, 3], scope='AC_conv1')
                AC = slim.max_pool2d(AC, [2, 2], scope='AC_pool2')
                AC = slim.conv2d(AC, 512, [3, 3], scope='AC_conv2')
                AC = slim.conv2d_transpose(AC, num_outputs=512,
                                           stride=4, kernel_size=[3, 3],
                                           padding='SAME',
                                           scope='AC_upsample')
                AC = tf.sigmoid(AC, 'sigmoid')
                GC = tf.multiply(AC, net)

        with tf.variable_scope('Classification'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):
                net = net + GL + GC
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')

                # predict category
                net_category = slim.conv2d(net, category_classes, [1, 1], scope='fc8_category')
                net_category = tf.squeeze(net_category, [1, 2], name='fc8_category/squeezed')
                #net_category = tf.nn.softmax(net_category, name='Predictions_category')
                #net_category = tf.layers.dense(net_category, category_classes, name='Predictions_category')

                # predict attribute
                net_attribute = slim.conv2d(net, attribute_classes, [1, 1], activation_fn=tf.nn.sigmoid, scope='fc8_attribute')
                net_attribute = tf.squeeze(net_attribute, [1, 2], name='fc8_attribute/squeezed')
                #net_attribute = tf.layers.dense(net_attribute, attribute_classes, activation=None, name='Predictions_attribute')

        return net_category, net_attribute

    def BCRNNBlock(self, heat_maps, maps_num, maps_idxs, scope):
        with tf.variable_scope(scope):
            if maps_num == 2:
                grammar_serial = tf.stack([heat_maps[maps_idxs[0]],
                                           heat_maps[maps_idxs[1]]], axis=3)
            else:
                grammar_serial = tf.stack([heat_maps[maps_idxs[0]],
                                           heat_maps[maps_idxs[1]],
                                           heat_maps[maps_idxs[2]]], axis=3)
            # grammar_serial_RK1 shape (batch_size, time_steps, row, col)
            grammar_serial = tf.transpose(grammar_serial, (0, 3, 1, 2))
            grammar_serial = tf.expand_dims(grammar_serial, 4)
            refined_heatmaps = self.multiLayerBidirectionalRnn(1, 3, grammar_serial, [maps_num])
            refined_heatmaps = tf.squeeze(refined_heatmaps, [4])
            refined_heatmaps = tf.transpose(refined_heatmaps, (1, 0, 2, 3))
            return refined_heatmaps

    def multiLayerBidirectionalRnn(self, num_units, num_layers, inputs, seq_lengths):
        """multi layer bidirectional rnn
        Args:
            num_units: int, hidden unit of RNN cell
            num_layers: int, the number of layers
            inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
            seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths,
                        the length of the list is batch_size
        Returns:
            the output of last layer bidirectional rnn with concatenating
        """
        # TODO: add time_major parameter
        _inputs = inputs
        if len(_inputs.get_shape().as_list()) < 3:
            raise ValueError("the inputs must be 3-dimentional Tensor")
        batch_size = tf.shape(inputs)[0]

        for T in range(num_layers):
            # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
            # 恶心的很.如果不在这加的话,会报错的.
            with tf.variable_scope(None, default_name="BCRNN_" + str(T)):
                # rnn_cell_fw = CRNN(2, [28, 28, 1], 1, [2, 2])
                # rnn_cell_bw = CRNN(2, [28, 28, 1], 1, [2, 2])

                kwarg = {'input_shape': [28, 28, 1], 'output_channels': 1, 'kernel_shape': [3, 3]}
                rnn_cell_fw = tf.contrib.rnn.Conv2DLSTMCell('conv_2d_lstm_cell_fw', **kwarg)
                rnn_cell_bw = tf.contrib.rnn.Conv2DLSTMCell('conv_2d_lstm_cell_bw', **kwarg)

                initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
                initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
                output, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                _inputs,
                                                                initial_state_fw=initial_state_fw,
                                                                initial_state_bw=initial_state_bw,
                                                                dtype=tf.float32,
                                                                scope="BCRNN_" + str(T))
                # output, state = tf.nn.static_bidirectional_rnn(rnn_cell_fw, rnn_cell_bw,
                #                                                _inputs, sequence_length=seq_lengths,
                #                                                initial_state_fw=initial_state_fw,
                #                                                initial_state_bw=initial_state_bw,
                #                                                dtype=tf.float32,
                #                                                scope="BCRNN_" + str(T))
                # generate input for next bcrnn layer
                # _inputs = tf.concat(output, 2)
                output_fw, output_bw = output[0], output[1]
                # _inputs shape (batch_size, time_steps, row, col)
                _inputs = _inputs + output_fw + output_bw

        return _inputs


if __name__ == '__main__':
    a = tf.constant([[2, 2], [2, 2]])
    b = tf.constant([[1, 2, 3], [3, 4, 5]])

    # c = tf.reduce_max([a[0], b[1]], axis=1)
    with tf.Session() as sess:
        b_result = sess.run(b[1])
        print(b_result, b.shape)
