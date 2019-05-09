#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: data_utils.py
@time: 2019/4/24 18:26
@desc:
"""

import os
import re
import sys
import math
import random
import threading
from queue import Queue
from datetime import datetime
import numpy as np
import tensorflow as tf

from hparams import hparams as hp

slim = tf.contrib.slim


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureLists(feature=[_int64_feature(v) for v in values])


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureLists(feature=[_bytes_feature(v) for v in values])


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _convert_image_example(image_string, image_format, category, attribute, height, width, depth):
    """Build an Example proto for an example

    :param image_string: string, JPEG encoding of RGB image.
    :param image_format: string, format of the encoding image.
    :param category: int, the index of image label.
    :param attribute: list, the
    :param height:
    :param width:
    :param depth:
    :return:
    """
    context_feature = {
        'image_raw': _bytes_feature(image_string),
        'image_format': _bytes_feature(image_format),
        'category': _int64_feature(category),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
    }

    sequence_feature = {
        'attribute': _int64_feature_list(attribute),
    }

    return tf.train.SequenceExample(context=context_feature, feature_list=sequence_feature)


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg()

    def png_to_jpeg(self, image_data):
        # convert the image from png to jpeg
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def read_image_dims(self, image_data):
        image = self._decode_jpeg(self._sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    :param filename: string, path of the image file
    :return: boolean, indicating if the image is a PNG.
    """
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.png'


def _process_image(filename, coder):
    """Process a single image file

    :param filename: string, path to an image file  e.g.: '/path/to/image.jpg'.
    :param coder: instance of ImageReader to provide Tensorflow image coding utils.
    :return:
        image_data: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Clean the dirty data.
    if _is_png(filename):
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check the image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def process_image_files_batch(coder, thread_index, ranges, name, output_directory,
                              dataset, num_shards, store_image, error_queue):
    """Processes and save list of images as TFRecord in 1 thread.

    :param coder: instance of ImageReader to provide TensorFlow image coding utils.
    :param thread_index: integer, unique batch to run index is within [0, len(ranges)].
    :param ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
    :param name: string, unique identifier specifying the data set (e.g. `train` or `test`).
    :param output_directory: string, file path to store the tf records.
    :param dataset: list, a list of image example dicts.
    :param num_shards: integer, the number of shards for this data set.
    :param store_image: boolean, should the image be stored in TFRecord.
    :param error_queue: Queue, a queue to place image examples that failed.
    :return: None
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    error_counter = 0

    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name. e.g. 'train-0001-of-0010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file_path = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file_path)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            image_example = dataset[i]
            filename = str(image_example['filename'])


            try:
                if store_image:
                    image_buffer = image_example['encoded']
                else:
                    image_buffer = ''
                height = image_example['height']
                width = image_example['width']
                image_format = image_example['image_format']
                num_channels = image_example['channels']
                category = image_example['category']
                attribute = image_example['attribute']
                example = _convert_image_example(image_buffer, image_format, category,
                                                 attribute, height, width, num_channels)

                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1
            except Exception as e:
                error_counter += 1
                error_msg = repr(e)
                image_example['error_msg'] = error_msg
                error_queue.put(image_example)

            if not counter % 1000:
                print('%s [thread %d]: Processd %d of %d images in thread batch, with %d errors.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
                sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %s, with %d errors.' %
              (datetime.now(), thread_index, shard_counter, output_filename, error_counter))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %d shards, with %d errors.' %
          (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
    sys.stdout.flush()
