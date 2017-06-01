from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.training import saver as saver_lib


def check_input_checkpoint(input_checkpoint):
    if not saver_lib.checkpoint_exists(input_checkpoint):
        print("Input checkpoint '{}' doesn't exist!".format(input_checkpoint))
        exit(-1)


def restore_from_checkpoint(sess, input_checkpoint):
    saver = tf.train.import_meta_graph('{}.meta'.format(input_checkpoint))
    saver.restore(sess, input_checkpoint)
    return saver
