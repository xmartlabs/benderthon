#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility to freeze TensorFlow graphs."""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.tools import freeze_graph

from benderthon.util import check_input_checkpoint, restore_from_checkpoint


def _output_node_names_string_as_list(output_node_names):
    """Return a list of containing output_node_names if it's a string, otherwise return just output_node_names."""
    if type(output_node_names) is unicode or type(output_node_names) is str:
        return [output_node_names]
    else:
        return output_node_names


def freeze_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
    """Freeze and shrink the graph based on a checkpoint and the output node names."""
    check_input_checkpoint(input_checkpoint)

    output_node_names = _output_node_names_string_as_list(output_node_names)

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

    output_node_names = _output_node_names_string_as_list(output_node_names)

    with tf.Session() as sess:
        restore_from_checkpoint(sess, input_checkpoint)
        save_graph_only(sess, output_file_path, output_node_names)

# def save_weights_only_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
#     """Save the weight values of the checkpoints in a format that can be read by Bender."""
#     _check_input_checkpoint(input_checkpoint)
#
#     output_node_names = _output_node_names_string_as_list(output_node_names)
#
#     with tf.Session() as sess:
#         _restore_from_checkpoint(sess, input_checkpoint)
#
#         saver = tf.train.Saver()
#         saver.save(sess, 'a', write_meta_graph=False, write_state=False)
#
#         # TODO
