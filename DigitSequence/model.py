"""The models used to classify both the synthetic data created from MNIST and the SVHN dataset.

"""

from __future__ import division
from __future__ import print_function


import tensorflow as tf

import layers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('kernel_size', 5,
                            """Size of the patch.""")
tf.app.flags.DEFINE_integer('depth', 64,
                            """Depth of layer.""")
tf.app.flags.DEFINE_integer('num_hidden', 128,
                            """Size of hidden layer.""")


def svhn_model(data):

    # block1
    with tf.variable_scope('conv1') as scope:
        conv1 = layers.conv2d(data, FLAGS.depth, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    with tf.variable_scope('conv2') as scope:
        conv2 = layers.conv2d(conv1, FLAGS.depth, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    pool1 = layers.max_pool(conv2,kernel_size=FLAGS.kernel_size, stride=2, padding='VALID')

    # block2
    with tf.variable_scope('conv3') as scope:
        conv3 = layers.conv2d(pool1, FLAGS.depth*2, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    with tf.variable_scope('conv4') as scope:
        conv4 = layers.conv2d(conv3, FLAGS.depth*2, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    pool2 = layers.max_pool(conv4, kernel_size=FLAGS.kernel_size, stride=2, padding='VALID')

    # block3
    with tf.variable_scope('conv5') as scope:
        conv5 = layers.conv2d(pool2, FLAGS.depth*4, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    with tf.variable_scope('conv6') as scope:
        conv6 = layers.conv2d(conv5, FLAGS.depth*4, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    with tf.variable_scope('conv7') as scope:
        conv7 = layers.conv2d(conv6, FLAGS.depth*4, kernel_size=FLAGS.kernel_size, stride=1, padding='SAME', scope=scope)
    pool3 = layers.max_pool(conv7, kernel_size=FLAGS.kernel_size, stride=2, padding='VALID')

    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool3, [FLAGS.BATCH_SIZE, -1])
        #dim = reshape.get_shape()[1].value

    # FC layers
    with tf.variable_scope('fc0') as scope:
        fc0 = layers.fc(reshape, FLAGS.num_hidden, activation=tf.nn.relu,
       bias=0.0, scope=scope)
    with tf.variable_scope('fc1') as scope:
        fc1 = layers.fc(fc0, FLAGS.num_hidden, activation=tf.nn.relu,
       bias=0.0, scope=scope)

    # logits
    with tf.variable_scope('softmax0') as scope:
        logits0 = layers.fc(fc1, 5, activation=None,
       bias=0.0, scope=scope)
    with tf.variable_scope('softmax1') as scope:
        logits1 = layers.fc(fc1, 11, activation=None,
       bias=0.0, scope=scope)
    with tf.variable_scope('softmax2') as scope:
        logits2 = layers.fc(fc1, 11, activation=None,
       bias=0.0, scope=scope)
    with tf.variable_scope('softmax3') as scope:
        logits3 = layers.fc(fc1, 11, activation=None,
       bias=0.0, scope=scope)
    with tf.variable_scope('softmax4') as scope:
        logits4 = layers.fc(fc1, 11, activation=None,
       bias=0.0, scope=scope)
    with tf.variable_scope('softmax5') as scope:
        logits5 = layers.fc(fc1, 11, activation=None,
       bias=0.0, scope=scope)

    return [logits0, logits1, logits2, logits3, logits4, logits5]
