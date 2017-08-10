""" This file contains typical Neural Network layers to be used later.

    If the behavior of a layer changes between training and testing, an argument is_training is used.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def conv2d(inputs, num_filters_out, kernel_size, stride=1,
           padding='SAME', activation=tf.nn.relu, bias=0.0,
           trainable=True, scope=None, reuse=None):

  """ Adds a 2D convolution layer.

  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input, and a second variable called 'biases'
  added to the result of the convolution.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
    a tensor representing the output of the operation.
  """
  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
    stride_h, stride_w = stride, stride
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_h, kernel_w, num_filters_in, num_filters_out]
    weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    l2_regularizer = None

    weights = tf.get_variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable)

    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)

    bias_shape = [num_filters_out,]
    bias_initializer = tf.constant_initializer(bias)
    biases = tf.get_variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable)

    outputs = tf.nn.bias_add(conv, biases)

    if activation:
      outputs = activation(outputs)
    return outputs

def fc(inputs, num_units_out, activation=tf.nn.relu,
       bias=0.0, trainable=True,  scope=None, reuse=None):

  """Adds a fully connected layer.

  FC creates a variable called 'weights', representing the fully connected
  weight matrix, that is multiplied by the input, plus a
  second variable called 'biases' is added to the result of the initial
  vector-matrix multiplication.

  Args:
    inputs: a [B x N] tensor where B is the batch size and N is the number of
            input units in the layer.
    num_units_out: the number of output units in the layer.
    activation: activation function.
    stddev: the standard deviation for the weights.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
     the tensor variable representing the result of the series of operations.
  """
  with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    num_units_in = inputs.get_shape()[1]
    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    l2_regularizer = None
    weights = tf.get_variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable)

    bias_shape = [num_units_out,]
    bias_initializer = tf.constant_initializer(bias)
    biases = tf.get_variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable)

    outputs = tf.nn.xw_plus_b(inputs, weights, biases)

    if activation:
      outputs = activation(outputs)
    return outputs

def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):

  """Adds a Max Pooling layer.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: a list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the results of the pooling operation.

  Raises:
    ValueError: if 'kernel_size' is not a 2-D list
  """
  with tf.name_scope(scope, 'MaxPool', [inputs]):
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
    stride_h, stride_w = stride, stride
    return tf.nn.max_pool(inputs, ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1], padding=padding)

def dropout(inputs, keep_prob=0.5, is_training=True, scope=None):

  """Returns a dropout layer applied to the input.

  Args:
    inputs: the tensor to pass to the Dropout layer.
    keep_prob: the probability of keeping each input unit.
    is_training: whether or not the model is in training mode. If so, dropout is
    applied and values scaled. Otherwise, inputs is returned.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the output of the operation.
  """
  if is_training and keep_prob > 0:
    with tf.name_scope(scope, 'Dropout', [inputs]):
      return tf.nn.dropout(inputs, keep_prob)
  else:
    return inputs




