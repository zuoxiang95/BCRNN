#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zuoxiang@jd.com
@file: build_tf_records.py
@time: 2019/4/18 17:31
@desc:
"""

import re
import sys
import random
import threading

import numpy as np
import tensorflow as tf

from queue import Queue
from datetime import datetime
from hparams import hparams as hp
from data_utils import ImageReader, process_image_files_batch


slim = tf.contrib.slim


def create_tfrecord(dataset, dataset_name, output_directory, num_shards,
                    num_threads, shuffle=True, store_image=True):
    """Create TFRecords

    :param dataset: list, a list of an image json.
    :param dataset_name:
    :param output_directory:
    :param num_shards:
    :param num_threads:
    :param shuffle:
    :param store_image:
    :return:
    """
    # Images in TFRecords set must be shuffled properly
    if shuffle:
        random.shuffle(dataset)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(dataset), num_shards+1).astype(int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Lanching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image reader.
    image_reader = ImageReader()

    # A Queue to hold the image examples that fail to process.
    error_queue = Queue()

    for thread_index in range(len(ranges)):
        args = (image_reader, thread_index, ranges, dataset_name, output_directory,
                dataset, num_shards, store_image, error_queue)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(dataset)))

    # Collect the error messages.
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    print('%d examples failed.' % (len(errors),))

    return errors


def create_tf_records(input_file, save_file):
    with open(input_file, 'r') as f1:
        data = f1.readlines()

    iter = 0
    with tf.python_io.TFRecordWriter(save_file) as writer:
        for line in data:
            tmp_data = line.strip().split('\t')
            if len(tmp_data) != 3:
                raise AssertionError('Data split error! Please check data!')
            filenames = hp.image_path + tmp_data[0]
            with open(filenames, 'rb') as f2:
                encode_image = f2.read()
            category = int(tmp_data[1])
            attribute = re.split('\s+', tmp_data[2].strip())
            attribute = [int(i) for i in attribute]
            if len(attribute) != 1000:
                raise AssertionError("Attribute vector's shape not equal 1000! Please check data!")

            try:
                tf_example = _image_example(encode_image, category, attribute)
                iter += 1
            except Exception as e:
                raise e

            if iter % 500 == 0:
                print('Processed image num: {}'.format(iter))

            writer.write(tf_example.SerializeToString())
    print('Done!')


if __name__ == '__main__':
    create_tf_records(r'', r'')
