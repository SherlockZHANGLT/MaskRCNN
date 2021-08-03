#coding=utf-8
from __future__ import print_function
import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader



def read_and_decode(filename_queue, batch_size = 8, capacity=30, num_threads=1, min_after_dequeue=10, is_training=True):
    """和下面那个read_and_decode_for_lstm似乎没啥区别，就是features少了一个image/superpixels。"""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 读出来的那一个序列的样本
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channels': tf.FixedLenFeature([], tf.int64),       
        'image/image_data': tf.FixedLenFeature([], tf.string),
        'image/mask_class': tf.FixedLenFeature([], tf.string),
        'image/mask_instance': tf.FixedLenFeature([], tf.string)
        })
    height = tf.cast(features['image/height'], tf.int64)  # shape=()，一个数。
    width = tf.cast(features['image/width'], tf.int64)  # shape=()，一个数。
    image = tf.decode_raw(features['image/image_data'], tf.float32)  # shape=(?,)，那个?应该是代表批大小。现在是图像应该是给展平了。
    mask_class = tf.decode_raw(features['image/mask_class'], tf.uint8)  # shape=(?,)，现在是掩膜给展平了。
    mask_instance = tf.decode_raw(features['image/mask_instance'], tf.uint8)  # shape=(?,)，似乎也是掩膜给展平了，不知道为啥要再搞一个。。
    image = tf.reshape(image, [512, 512, 1])  # shape=(512, 512, 1)，现在变回那张图了。类型已经是tf.float32。
    mask_class=tf.reshape(mask_class, [512, 512])  # shape=(512, 512)，现在变回那张图的掩膜了。
    mask_class = tf.cast(mask_class,tf.int32)  # 现在把mask_class那个掩膜变成int32类型的数据。
    mask_instance= tf.reshape(mask_instance, [512, 512, 1])  # shape=(512, 512, 1) dtype=uint8 似乎也是掩膜。
    if is_training is True:
        image,mask_class,mask_instance = tf.train.shuffle_batch( [image,mask_class,mask_instance], batch_size=batch_size,
                                                 capacity=capacity,
                                                 num_threads=num_threads,
                                                 min_after_dequeue=min_after_dequeue)
    else:
        image,mask_class,mask_instance = tf.train.batch([image,mask_class,mask_instance], batch_size=batch_size,num_threads=num_threads,capacity=capacity)
        # 上句，三个东西的shape分别为(1, 512, 512, 1)（这是因为batch_size给的是1啊）、(1, 512, 512)、(1, 512, 512, 1)。
    return image, mask_class, mask_instance

def read_and_decode_for_lstm(filename_queue, batch_size = 8, capacity=30, num_threads=1, min_after_dequeue=10, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channels': tf.FixedLenFeature([], tf.int64),       
        'image/image_data': tf.FixedLenFeature([], tf.string),
        'image/superpixels': tf.FixedLenFeature([], tf.string),  # 比不带for_lstm的多了这个东西
        'image/mask_class': tf.FixedLenFeature([], tf.string),
        'image/mask_instance': tf.FixedLenFeature([], tf.string)
        })
    """上面那个feature似乎也是个字符串下标的数组，应该是说图像长宽、通道数、类别等信息的。"""
    height = tf.cast(features['image/height'], tf.int64)  # shape=()，一个数，和不带_for_lstm的一样。
    width = tf.cast(features['image/width'], tf.int64)  # shape=()，一个数，和不带_for_lstm的一样。
    image = tf.decode_raw(features['image/image_data'], tf.int16)  # shape=(?,)，dtype=int16。（不带for_lstm的是tf.float32，不过后面也变成了float32。）
    superpixles = tf.decode_raw(features['image/superpixels'], tf.int64)  # shape=(?,)，比不带for_lstm的多了这个东西
    mask_class = tf.decode_raw(features['image/mask_class'], tf.uint8)  # shape=(?,)，现在是掩膜给展平了，和不带_for_lstm的一样。
    mask_instance = tf.decode_raw(features['image/mask_instance'], tf.uint8)  # shape=(?,)，似乎也是掩膜给展平了，和不带_for_lstm的一样，不知道为啥要再搞一个。。
    image = tf.reshape(image, [512, 512, 1])  # shape=(512, 512, 1)，现在变回那张图了。类型已经是tf.int16。
    image = tf.cast(image,tf.float32)  # 那张图类型变成tf.float32，到这儿，图片和不带for_lstm的一样了。
    superpixles=tf.reshape(superpixles, [512, 512])  # 比不带for_lstm的多了这个东西
    mask_class=tf.reshape(mask_class, [512, 512])  # shape=(512, 512)，现在变回那张图的掩膜了。
    mask_class = tf.cast(mask_class,tf.int32)  # 现在把mask_class那个掩膜变成int32类型的数据。
    mask_instance= tf.reshape(mask_instance, [512, 512, 1])  # shape=(512, 512, 1)，这个在后面没用。。
    """以上几句就是把那个feature还原成图像、超像素等等那六个信息"""
    if is_training is True:
        """
        发现这个is_training与否的区别，就是弄批次的时候是否用乱序（shuffle_batch还是batch）
        而，正是因为用了那个shuffle_batch，所以才会有个min_after_dequeue参数，控制乱序的混乱程度。
        """
        image,mask_class,superpixles = tf.train.shuffle_batch( [image,mask_class,superpixles], batch_size=batch_size,  # 把不带for_lstm的mask_instance换成了superpixles
                                                 capacity=capacity,
                                                 num_threads=num_threads,
                                                 min_after_dequeue=min_after_dequeue)
    else:
        image,mask_class,superpixles = tf.train.batch([image,mask_class,superpixles], batch_size=batch_size,\
                                                        num_threads=num_threads,capacity=capacity)
        # 上句，三个东西的shape分别为(1, 512, 512, 1)（这是因为batch_size给的是1啊）、(1, 512, 512)、(1, 512, 512, 1)。
    return image, mask_class, superpixles  # 返回的是图像本身、标签类别、超像素。


def reshape_list(l, shape=None):
    # 把list（列表）变形。没有用到，不细看了。
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

# =========================================================================== #
# Training utils.
# =========================================================================== #
def print_configuration(flags, ssd_params, data_sources, save_dir=None):
    # 打印训练参数。没有用到，不细看了。
    """Print the training configuration.
    """
    def print_config(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# SSD net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(dict(ssd_params._asdict()), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    print_config(None)
    # Save to a text file as well.
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, 'training_config.txt')
        with open(path, "w") as out:
            print_config(out)
            
def configure_learning_rate(flags, num_samples_per_epoch, global_step):
    # 处理学习率的事情
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch / flags.batch_size *
                      flags.num_epochs_per_decay)
    """
    以下就是根据输入的学习率衰减方式，返回不同的学习率
        其中，tf.train.exponential_decay的理解是：
        它返回一个标量张量（应该是个指数衰减的序列吧），官网（https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay）说是：
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    多项式衰减的那个类似理解。
    """
    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          flags.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         flags.learning_rate_decay_type)

def configure_optimizer(flags, learning_rate):
    # 返回一个优化子，然后在主函数中，这个优化子定义一个操作（op）
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=flags.adam_beta1,
            beta2=flags.adam_beta2,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer


def get_init_fn(flags):
    # 似乎是训练暖启动的函数。没有用到，不细看了。
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
      An init function run by the supervisor.
    """
    if flags.checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(flags.model_name,
                                 flags.checkpoint_model_scope): var
             for var in variables_to_restore}


    if tf.gfile.IsDirectory(flags.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
    else:
        checkpoint_path = flags.checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
    # 获取待训练参数的函数。没有用到，不细看了。
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train