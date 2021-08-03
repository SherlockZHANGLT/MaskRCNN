#coding=utf-8
from __future__ import print_function
import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

def configure_learning_rate(batch_size, num_epochs_per_decay,
                            original_learning_rate, learning_rate_decay_type, learning_rate_decay_factor, end_learning_rate,
                            num_samples_per_epoch, global_step):
    # 处理学习率的事情
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch / batch_size *
                      num_epochs_per_decay)
    """
    以下就是根据输入的学习率衰减方式，返回不同的学习率
        其中，tf.train.exponential_decay的理解是：
        它返回一个标量张量（应该是个指数衰减的序列吧），官网（https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay）说是：
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    多项式衰减的那个类似理解。
    """
    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(original_learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(original_learning_rate, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(original_learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         learning_rate_decay_type)