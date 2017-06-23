#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import imp
import os

import tensorflow as tf

from benderthon import tf_freeze, util


def caffe_freeze(caffe_def_path, caffemodel_path, inputs, output_file_path, output_node_names):
    try:
        # noinspection PyUnresolvedReferences
        from caffeflow import convert
    except ImportError:
        raise Exception("caffe-tensorflow package needs to be installed to freeze Caffe models. Check out the README.")

    output_node_names = util.output_node_names_string_as_list(output_node_names)

    with util.TemporaryDirectory() as temp_dir_path:
        params_values_output_path = os.path.join(temp_dir_path, 'params_values.npy')
        network_output_path = os.path.join(temp_dir_path, 'network.py')

        convert.convert(caffe_def_path, caffemodel_path, params_values_output_path, network_output_path, False)

        network_module = imp.load_source('module.name', network_output_path)

        network = network_module.Graph(inputs)

        with tf.Session() as sess:
            network.load(params_values_output_path, sess)

            saver = tf.train.Saver()

            checkpoint_path = os.path.join(temp_dir_path, 'pose.ckpt')
            saver.save(sess, checkpoint_path)

        tf_freeze.freeze_from_checkpoint(checkpoint_path, output_file_path, output_node_names)
