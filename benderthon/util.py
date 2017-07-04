# -*- coding: utf-8 -*-
"""Util functions used by Bender subcommands."""

from __future__ import absolute_import, division, print_function, unicode_literals

import shutil
import tempfile
import warnings

from six import string_types
import tensorflow as tf
from tensorflow.python.training import saver as saver_lib


def check_input_checkpoint(input_checkpoint):
    """Check if input_checkpoint is a valid path or path prefix."""
    if not saver_lib.checkpoint_exists(input_checkpoint):
        print("Input checkpoint '{}' doesn't exist!".format(input_checkpoint))
        exit(-1)


def restore_from_checkpoint(sess, input_checkpoint):
    """Return a TensorFlow saver from a checkpoint containing the metagraph."""
    saver = tf.train.import_meta_graph('{}.meta'.format(input_checkpoint))
    saver.restore(sess, input_checkpoint)
    return saver


def output_node_names_string_as_list(output_node_names):
    """Return a list of containing output_node_names if it's a string, otherwise return just output_node_names."""
    if isinstance(output_node_names, string_types):
        return [output_node_names]
    else:
        return output_node_names


class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.

    Inspired from https://hg.python.org/cpython/file/3.6/Lib/tempfile.py
    """

    def __init__(self, suffix='', prefix='', dir_=None):
        self.name = tempfile.mkdtemp(suffix, prefix, dir_)

    @classmethod
    def _cleanup(cls, name, warn_message):
        shutil.rmtree(name)
        warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        shutil.rmtree(self.name)
