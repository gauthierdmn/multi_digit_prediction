

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from datetime import datetime

from model import svhn_model
from loss import cross_entropy_loss
import preprocessing

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('MAX_STEPS', 3001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('IM_SIZE', 32,
                            """Size of cropped SVHN images.""")


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == labels)
          / len(predictions))

def train_svhn():
  """Train SVHN data for a number of steps."""
  graph = tf.Graph()
  with graph.as_default():

    # Get images and labels for SVHN.
    train_dataset, test_dataset, train_labels, \
    test_labels, train_lengths, test_lengths = preprocessing.load_svhn()

    # Get images and labels for CIFAR-10.
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(FLAGS.BATCH_SIZE, FLAGS.IM_SIZE,
                                                         FLAGS.IM_SIZE, FLAGS.num_channels))
    tf_train_lengths = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    tf_train_labels1 = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    tf_train_labels2 = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    tf_train_labels3 = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    tf_train_labels4 = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    tf_train_labels5 = tf.placeholder(tf.int32, shape=(FLAGS.BATCH_SIZE))
    #tf_test_dataset = tf.constant(test_dataset)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits0, logits1, logits2, logits3, logits4, logits5 = svhn_model(tf_train_dataset)
    train_predictions0, train_predictions1, train_predictions2, train_predictions3, train_predictions4, train_predictions5 = [tf.nn.softmax(logits0), tf.nn.softmax(logits1), tf.nn.softmax(logits2),
    tf.nn.softmax(logits3), tf.nn.softmax(logits4), tf.nn.softmax(logits5)]

    # Calculate loss.
    loss = cross_entropy_loss(logits0, tf_train_lengths) + cross_entropy_loss(logits1, tf_train_labels1) + cross_entropy_loss(logits2, tf_train_labels2) + cross_entropy_loss(logits3, tf_train_labels3) + cross_entropy_loss(logits4, tf_train_labels4) + cross_entropy_loss(logits5, tf_train_labels5)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 7500, 0.5, staircase=True)
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

  with tf.Session(graph=graph):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('Initialized')

    for step in range(FLAGS.MAX_STEPS):
        start_time = time.time()

        offset = (step * FLAGS.BATCH_SIZE) % (train_labels.shape[0] - FLAGS.BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + FLAGS.BATCH_SIZE), :, :, :]
        batch_lengths = train_lengths[offset:(offset + FLAGS.BATCH_SIZE)]
        batch_labels1 = train_labels[offset:(offset + FLAGS.BATCH_SIZE), 0]
        batch_labels2 = train_labels[offset:(offset + FLAGS.BATCH_SIZE), 1]
        batch_labels3 = train_labels[offset:(offset + FLAGS.BATCH_SIZE), 2]
        batch_labels4 = train_labels[offset:(offset + FLAGS.BATCH_SIZE), 3]
        batch_labels5 = train_labels[offset:(offset + FLAGS.BATCH_SIZE), 4]

        feed_dict = {tf_train_dataset: batch_data, tf_train_lengths: batch_lengths, tf_train_labels1: batch_labels1, tf_train_labels2: batch_labels2, tf_train_labels3: batch_labels3, tf_train_labels4: batch_labels4, tf_train_labels5: batch_labels5}

        _, loss_value, predictions0, predictions1, predictions2, predictions3, predictions4, predictions5  = sess.run([optimizer,
                                                                     loss, train_predictions0,
                                                                     train_predictions1,
                                                                     train_predictions2,
                                                                     train_predictions3,
                                                                     train_predictions4,
                                                                     train_predictions5], feed_dict=feed_dict)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 50 == 0:
            accuracy_batch = ((accuracy(predictions0,batch_lengths)
                                  + accuracy(predictions1,batch_labels1)
                                  + accuracy(predictions2,batch_labels2)
                                  + accuracy(predictions3,batch_labels3)
                                  + accuracy(predictions4,batch_labels4)
                                  + accuracy(predictions5,batch_labels5))/6)

            format_str = ('%s: step %d, loss = %.2f, batch accuracy = %.1f%% (%.3f ''sec/batch)')
            print(format_str % (datetime.now(), step, loss_value, accuracy_batch, duration))

if __name__ == '__main__':
    train_svhn()




