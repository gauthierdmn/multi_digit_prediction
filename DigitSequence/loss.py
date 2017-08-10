""" In this script is defined the cross entropy loss function and (TODO) L1 and L2 regularization

"""

import tensorflow as tf

def cross_entropy_loss(logits, labels):

    """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

    Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.

    Returns:
    A tensor with the softmax_cross_entropy loss.
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss