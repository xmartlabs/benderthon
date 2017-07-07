#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility to freeze TensorFlow graphs."""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.tools import freeze_graph

from benderthon.util import check_input_checkpoint, output_node_names_string_as_list, restore_from_checkpoint


def freeze_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
    """Freeze and shrink the graph based on a checkpoint and the output node names."""
    check_input_checkpoint(input_checkpoint)

    output_node_names = output_node_names_string_as_list(output_node_names)

    with tf.Session() as sess:
        restore_from_checkpoint(sess, input_checkpoint)
        freeze_graph.freeze_graph_with_def_protos(input_graph_def=sess.graph_def, input_saver_def=None,
                                                  input_checkpoint=input_checkpoint,
                                                  output_node_names=','.join(output_node_names),
                                                  restore_op_name='save/restore_all',
                                                  filename_tensor_name='save/Const:0', output_graph=output_file_path,
                                                  clear_devices=True, initializer_nodes='')


def save_graph_only(sess, output_file_path, output_node_names):
    """Save a small version of the graph based on a session and the output node names."""
    for node in sess.graph_def.node:
        node.device = ''
    graph_def = graph_util.extract_sub_graph(sess.graph_def, output_node_names)
    output_dir, output_filename = os.path.split(output_file_path)
    graph_io.write_graph(graph_def, output_dir, output_filename, as_text=True)


def save_graph_only_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
    """Save a small version of the graph based on a checkpoint and the output node names."""
    check_input_checkpoint(input_checkpoint)

    output_node_names = output_node_names_string_as_list(output_node_names)

    with tf.Session() as sess:
        restore_from_checkpoint(sess, input_checkpoint)
        save_graph_only(sess, output_file_path, output_node_names)


def save_weights(sess, output_path, conv_var_names=None, conv_transpose_var_names=None):
    """Save the weights of the trainable variables, each one in a different file in output_path."""
    if not conv_var_names:
        conv_var_names = []

    if not conv_transpose_var_names:
        conv_transpose_var_names = []

    for var in tf.trainable_variables():
        filename = '{}-{}'.format(output_path, var.name.replace(':', '-').replace('/', '-'))

        if var.name in conv_var_names:
            var = tf.transpose(var, perm=[3, 0, 1, 2])
        elif var.name in conv_transpose_var_names:
            var = tf.transpose(var, perm=[3, 1, 0, 2])

        value = sess.run(var)

        # noinspection PyTypeChecker
        with open(filename, 'w') as file_:
            value.tofile(file_)


def save_weights_from_checkpoint(input_checkpoint, output_path, conv_var_names=None, conv_transpose_var_names=None):
    """Save the weights of the trainable variables given a checkpoint, each one in a different file in output_path."""
    check_input_checkpoint(input_checkpoint)

    with tf.Session() as sess:
        restore_from_checkpoint(sess, input_checkpoint)
        save_weights(sess, output_path, conv_var_names=conv_var_names,
                     conv_transpose_var_names=conv_transpose_var_names)
