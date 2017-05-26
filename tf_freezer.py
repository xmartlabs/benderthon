#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_io, graph_util
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib


def _output_node_names_string_as_list(output_node_names):
    if type(output_node_names) is str:
        return [output_node_names]
    else:
        return output_node_names


def _check_input_checkpoint(input_checkpoint):
    if not saver_lib.checkpoint_exists(input_checkpoint):
        print("Input checkpoint '{}' doesn't exist!".format(input_checkpoint))
        exit(-1)


def _restore_from_checkpoint(sess, input_checkpoint):
    saver = tf.train.import_meta_graph('{}.meta'.format(input_checkpoint))
    saver.restore(sess, input_checkpoint)
    return saver


def freeze_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
    _check_input_checkpoint(input_checkpoint)

    output_node_names = _output_node_names_string_as_list(output_node_names)

    with tf.Session() as sess:
        _restore_from_checkpoint(sess, input_checkpoint)
        freeze_graph.freeze_graph_with_def_protos(input_graph_def=sess.graph_def, input_saver_def=None,
                                                  input_checkpoint=input_checkpoint,
                                                  output_node_names=','.join(output_node_names),
                                                  restore_op_name='save/restore_all',
                                                  filename_tensor_name='save/Const:0', output_graph=output_file_path,
                                                  clear_devices=True, initializer_nodes='')


def save_graph_only(sess, output_file_path, output_node_names):
    for node in sess.graph_def.node:
        node.device = ''
    graph_def = graph_util.extract_sub_graph(sess.graph_def, output_node_names)
    output_dir, output_filename = os.path.split(output_file_path)
    graph_io.write_graph(graph_def, output_dir, output_filename, as_text=False)


def save_graph_only_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
    _check_input_checkpoint(input_checkpoint)

    output_node_names = _output_node_names_string_as_list(output_node_names)

    with tf.Session() as sess:
        _restore_from_checkpoint(sess, input_checkpoint)
        save_graph_only(sess, output_file_path, output_node_names)

# def save_weights_only_from_checkpoint(input_checkpoint, output_file_path, output_node_names):
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


def _main():
    parser = argparse.ArgumentParser(
        description="Utility to easily convert TensorFlow checkpoints into minimal frozen graphs in binary protobuf "
                    "format.")
    parser.add_argument('input_checkpoint', help="checkpoint path to load")
    parser.add_argument('output_file', help="file to save the binary protobuf graph")
    parser.add_argument('output_node_names', help="the name of the output nodes, comma separated")
    parser.add_argument('--no-weights', action='store_true',
                        help="indicate that the variables are not converted to consts")
    args = parser.parse_args()
    if args.no_weights:
        save_graph_only_from_checkpoint(args.input_checkpoint, args.output_file, args.output_node_names)
    else:
        freeze_from_checkpoint(args.input_checkpoint, args.output_file, args.output_node_names)

if __name__ == '__main__':
    _main()
