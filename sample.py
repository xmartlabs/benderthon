#!/usr/bin/env python
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SEED = 24


if __name__ == '__main__':
    mnist = input_data.read_data_sets('data', one_hot=True, seed=SEED)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W_fc1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.01, seed=SEED))
    b_fc1 = tf.Variable(tf.zeros([1000]))
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    keep_prob1 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob1, seed=SEED)

    W_fc2 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.01, seed=SEED))
    b_fc2 = tf.Variable(tf.zeros([1000]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    keep_prob2 = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=keep_prob2, seed=SEED)

    W_fc3 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.01, seed=SEED))
    b_fc3 = tf.Variable(tf.zeros([10]))
    y = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3, name="Prediction")

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(1000):
            batch = mnist.train.next_batch(128)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob1: 0.8, keep_prob2: 0.5})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob1: 1.0,
                                            keep_prob2: 1.0})
        print("Test accuracy: {}".format(acc))

        saver.save(sess, 'checkpoints/mnist.ckpt')
