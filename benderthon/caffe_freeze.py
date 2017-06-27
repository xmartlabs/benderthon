#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import imp
import os

import tensorflow as tf

from benderthon import tf_freeze, util


def caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name='Graph'):
    try:
        # noinspection PyUnresolvedReferences
        from caffeflow import convert
    except ImportError:
        raise Exception("caffe-tensorflow package needs to be installed to freeze Caffe models. Check out the README.")

    with util.TemporaryDirectory() as temp_dir_path:
        params_values_output_path = os.path.join(temp_dir_path, 'params_values.npy')
        network_output_path = os.path.join(temp_dir_path, 'network.py')

        convert.convert(caffe_def_path, caffemodel_path, params_values_output_path, network_output_path, False)

        network_module = imp.load_source('module.name', network_output_path)
        network_class = getattr(network_module, graph_name)
        network = network_class(inputs)

        sess = tf.Session()

        network.load(params_values_output_path, sess)

        return sess


def freeze(caffe_def_path, caffemodel_path, inputs, output_file_path, output_node_names, graph_name='Graph'):
    with caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name=graph_name) as sess:
        saver = tf.train.Saver()

        with util.TemporaryDirectory() as temp_dir_path:
            checkpoint_path = os.path.join(temp_dir_path, 'pose.ckpt')
            saver.save(sess, checkpoint_path)

            output_node_names = util.output_node_names_string_as_list(output_node_names)

            tf_freeze.freeze_from_checkpoint(checkpoint_path, output_file_path, output_node_names)


def save_graph_only(caffe_def_path, caffemodel_path, inputs, output_file_path, output_node_names, graph_name='Graph'):
    with caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name=graph_name) as sess:
        tf_freeze.save_graph_only(sess, output_file_path, output_node_names)
