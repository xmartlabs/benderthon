#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility to freeze Caffe models."""

from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
import imp
import os

import tensorflow as tf

from benderthon import tf_freeze, util


@contextlib.contextmanager
def dummy_context_mgr(obj):
    yield obj


def caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name='Graph',
                                conversion_out_dir_path=None, use_padding_same=False):
    """Create a TensorFlow Session from a Caffe model."""
    try:
        # noinspection PyUnresolvedReferences
        from caffeflow import convert
    except ImportError:
        raise Exception("caffeflow package needs to be installed to freeze Caffe models. Check out the README file.")

    with (dummy_context_mgr(conversion_out_dir_path) or util.TemporaryDirectory()) as dir_path:
        params_values_output_path = os.path.join(dir_path, 'params_values.npy')
        network_output_path = os.path.join(dir_path, 'network.py')

        convert.convert(caffe_def_path, caffemodel_path, params_values_output_path, network_output_path, False,
                        use_padding_same=use_padding_same)

        network_module = imp.load_source('module.name', network_output_path)
        network_class = getattr(network_module, graph_name)
        network = network_class(inputs)

        sess = tf.Session()

        network.load(params_values_output_path, sess)

        return sess


def freeze(caffe_def_path, caffemodel_path, inputs, output_file_path, output_node_names, graph_name='Graph',
           conversion_out_dir_path=None, checkpoint_out_path=None, use_padding_same=False):
    """Freeze and shrink the graph based on a Caffe model, the input tensors and the output node names."""
    with caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name=graph_name,
                                     conversion_out_dir_path=conversion_out_dir_path,
                                     use_padding_same=use_padding_same) as sess:
        saver = tf.train.Saver()

        with (dummy_context_mgr(checkpoint_out_path) or util.TemporaryDirectory()) as temp_dir_path:
            checkpoint_path = checkpoint_out_path or os.path.join(temp_dir_path, 'pose.ckpt')
            saver.save(sess, checkpoint_path)

            output_node_names = util.output_node_names_string_as_list(output_node_names)

            tf_freeze.freeze_from_checkpoint(checkpoint_path, output_file_path, output_node_names)


def save_graph_only(caffe_def_path, caffemodel_path, inputs, output_file_path, output_node_names, graph_name='Graph',
                    use_padding_same=False):
    """Save a small version of the graph based on a Caffe model, the input tensors and the output node names."""
    with caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name=graph_name,
                                     use_padding_same=use_padding_same) as sess:
        tf_freeze.save_graph_only(sess, output_file_path, output_node_names)


def save_weights(caffe_def_path, caffemodel_path, inputs, output_path, graph_name='Graph', conv_var_names=None,
                 conv_transpose_var_names=None, use_padding_same=False):
    """Save the weights of the trainable variables, each one in a different file in output_path."""
    with caffe_to_tensorflow_session(caffe_def_path, caffemodel_path, inputs, graph_name=graph_name,
                                     use_padding_same=use_padding_same) as sess:
        tf_freeze.save_weights(sess, output_path, conv_var_names=conv_var_names,
                               conv_transpose_var_names=conv_transpose_var_names)
