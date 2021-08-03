import tensorflow as tf
"""https://www.v2ex.com/t/335186"""
def conv_1_1(input, output_channels, name):
    """1*1卷积，就是改变通道数用的。"""
    _, _, _, c = input.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w1', shape=[1, 1, c, output_channels], initializer=tf.contrib.keras.initializers.he_normal())
        b = tf.Variable(tf.constant(0.1, shape=[output_channels]))
        conv_and_biased = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME") + b
    return conv_and_biased

def inception_unit(inputdata, weights, biases):
    # A3 inception 3a
    inception_in = inputdata

    # Conv 1x1+S1
    inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1, 1, 1, 1], padding='SAME')
    inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
    inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
    # Conv 3x3+S1
    inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1, 1, 1, 1], padding='SAME')
    inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
    inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1, 1, 1, 1], padding='SAME')
    inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
    inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
    # Conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1, 1, 1, 1], padding='SAME')
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1, 1, 1, 1], padding='SAME')
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
    # MaxPool
    inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1, 1, 1, 1], padding='SAME')
    inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
    inception_MaxPool = tf.nn.relu(inception_MaxPool)
    # Concat
    # tf.concat(concat_dim, values, name='concat')
    # concat_dim 是 tensor 连接的方向（维度）， values 是要连接的 tensor 链表， name 是操作名。 cancat_dim 维度可以不一样，其他维度的尺寸必须一样。
    inception_out = tf.concat(values=[inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool], axis=3)
    return inception_out


def GoogleLeNet_topological_structure(x, weights, biases):
    pooling = {
        'pool1_3x3_S2': [1, 3, 3, 1],
        'pool2_3x3_S2': [1, 3, 3, 1],
        'pool3_3x3_S2': [1, 3, 3, 1],
        'pool4_3x3_S2': [1, 3, 3, 1]
    }

    # conv_W_3a = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 192, 64], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 192, 96], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 96, 128], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 192, 16], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 192, 32], stddev=0.1))
    # }
    conv_W_3a = {
        'inception_1x1_S1': tf.get_variable('w3a1', shape=[1, 1, 192, 64], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w3a2', shape=[1, 1, 192, 96], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w3a3', shape=[1, 1, 96, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w3a4', shape=[1, 1, 192, 16], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w3a5', shape=[5, 5, 16, 32], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w3a6', shape=[1, 1, 192, 32], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_3a = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape = [64])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape = [96])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape = [128])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape = [16])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape = [32])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape = [32]))
    }

    # conv_W_3b = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 256, 128], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 256, 128], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 128, 192], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 256, 32], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 32, 96], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 256, 64], stddev=0.1))
    # }
    conv_W_3b = {
        'inception_1x1_S1': tf.get_variable('w3b1', shape=[1, 1, 256, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w3b2', shape=[1, 1, 256, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w3b3', shape=[1, 1, 128, 192], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w3b4', shape=[1, 1, 256, 32], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w3b5', shape=[5, 5, 32, 96], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w3b6', shape=[1, 1, 256, 64], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_3b = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[192])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[32])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[96])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[64]))
    }

    # conv_W_4a = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 480, 192], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 480, 96], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 96, 208], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 480, 16], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 16, 48], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 480, 64], stddev=0.1))
    # }
    conv_W_4a = {
        'inception_1x1_S1': tf.get_variable('w4a1', shape=[1, 1, 480, 192], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w4a2', shape=[1, 1, 480, 96], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w4a3', shape=[1, 1, 96, 208], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w4a4', shape=[1, 1, 480, 16], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w4a5', shape=[5, 5, 16, 48], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w4a6', shape=[1, 1, 480, 64], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_4a = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[192])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[96])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[208])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[16])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[48])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[64]))
    }

    # conv_W_4b = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 512, 160], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 112], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 112, 224], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 24], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 24, 64], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 512, 64], stddev=0.1))
    # }
    conv_W_4b = {
        'inception_1x1_S1': tf.get_variable('w4b1', shape=[1, 1, 512, 160], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w4b2', shape=[1, 1, 512, 112], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w4b3', shape=[1, 1, 112, 224], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w4b4', shape=[1, 1, 512, 24], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w4b5', shape=[5, 5, 24, 64], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w4b6', shape=[1, 1, 512, 64], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_4b = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[160])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[112])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[224])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[24])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[64])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[64]))
    }

    # conv_W_4c = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 128, 256], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 24], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 24, 64], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 512, 64], stddev=0.1))
    # }
    conv_W_4c = {
        'inception_1x1_S1': tf.get_variable('w4c1', shape=[1, 1, 512, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w4c2', shape=[1, 1, 512, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w4c3', shape=[1, 1, 128, 256], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w4c4', shape=[1, 1, 512, 24], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w4c5', shape=[5, 5, 24, 64], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w4c6', shape=[1, 1, 512, 64], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_4c = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[256])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[24])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[64])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[64]))
    }

    # conv_W_4d = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 512, 112], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 144], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 144, 288], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 512, 32], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 512, 64], stddev=0.1))
    # }
    conv_W_4d = {
        'inception_1x1_S1': tf.get_variable('w4d1', shape=[1, 1, 512, 112], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w4d2', shape=[1, 1, 512, 144], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w4d3', shape=[1, 1, 144, 288], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w4d4', shape=[1, 1, 512, 32], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w4d5', shape=[5, 5, 32, 64], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w4d6', shape=[1, 1, 512, 64], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_4d = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[112])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[144])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[288])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[32])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[64])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[64]))
    }

    # conv_W_4e = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 528, 256], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 528, 160], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 160, 320], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 528, 32], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 32, 128], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 528, 128], stddev=0.1))
    # }
    conv_W_4e = {
        'inception_1x1_S1': tf.get_variable('w4e1', shape=[1, 1, 528, 256], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w4e2', shape=[1, 1, 528, 160], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w4e3', shape=[1, 1, 160, 320], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w4e4', shape=[1, 1, 528, 32], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w4e5', shape=[5, 5, 32, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w4e6', shape=[1, 1, 528, 128], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_4e = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[256])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[160])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[320])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[32])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[128]))
    }

    # conv_W_5a = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 832, 256], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 832, 160], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 160, 320], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 832, 32], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 32, 128], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 832, 128], stddev=0.1))
    # }
    conv_W_5a = {
        'inception_1x1_S1': tf.get_variable('w5a1', shape=[1, 1, 832, 256], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w5a2', shape=[1, 1, 832, 160], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w5a3', shape=[1, 1, 160, 320], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w5a4', shape=[1, 1, 832, 32], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w5a5', shape=[5, 5, 32, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w5a6', shape=[1, 1, 832, 128], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_5a = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[256])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[160])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[320])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[32])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[128]))
    }

    # conv_W_5b = {
    #     'inception_1x1_S1': tf.Variable(tf.truncated_normal([1, 1, 832, 384], stddev=0.1)),
    #     'inception_3x3_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 832, 192], stddev=0.1)),
    #     'inception_3x3_S1': tf.Variable(tf.truncated_normal([1, 1, 192, 384], stddev=0.1)),
    #     'inception_5x5_S1_reduce': tf.Variable(tf.truncated_normal([1, 1, 832, 48], stddev=0.1)),
    #     'inception_5x5_S1': tf.Variable(tf.truncated_normal([5, 5, 48, 128], stddev=0.1)),
    #     'inception_MaxPool': tf.Variable(tf.truncated_normal([1, 1, 832, 128], stddev=0.1))
    # }
    conv_W_5b = {
        'inception_1x1_S1': tf.get_variable('w5b1', shape=[1, 1, 832, 384], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1_reduce': tf.get_variable('w5b2', shape=[1, 1, 832, 192], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_3x3_S1': tf.get_variable('w5b3', shape=[1, 1, 192, 384], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1_reduce': tf.get_variable('w5b4', shape=[1, 1, 832, 48], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_5x5_S1': tf.get_variable('w5b5', shape=[5, 5, 48, 128], initializer=tf.contrib.keras.initializers.he_normal()),
        'inception_MaxPool': tf.get_variable('w5b6', shape=[1, 1, 832, 128], initializer=tf.contrib.keras.initializers.he_normal())
    }

    conv_B_5b = {
        'inception_1x1_S1': tf.Variable(tf.constant(0.1, shape =[384])),
        'inception_3x3_S1_reduce': tf.Variable(tf.constant(0.1, shape =[192])),
        'inception_3x3_S1': tf.Variable(tf.constant(0.1, shape =[384])),
        'inception_5x5_S1_reduce': tf.Variable(tf.constant(0.1, shape =[48])),
        'inception_5x5_S1': tf.Variable(tf.constant(0.1, shape =[128])),
        'inception_MaxPool': tf.Variable(tf.constant(0.1, shape =[128]))
    }


    # A0 输入数据
    _, h, w, c = x.get_shape().as_list()

    # A1  Conv 7x7_S2
    x = tf.nn.conv2d(x, weights['conv1_7x7_S2'], strides=[1, 2, 2, 1], padding='SAME')
    # 卷积层 卷积核 7*7 扫描步长 2*2
    x = tf.nn.bias_add(x, biases['conv1_7x7_S2'])
    # print (x.get_shape().as_list())  # [None, 256, 256, 64]
    # 偏置向量
    x = tf.nn.relu(x)
    # 激活函数
    x = tf.nn.max_pool(x, ksize=pooling['pool1_3x3_S2'], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 128, 128, 64)
    # 池化取最大值
    x = tf.nn.local_response_normalization(x, depth_radius=5 / 2.0, bias=2.0, alpha=1e-4, beta=0.75)
    # 局部响应归一化
    C1 = x

    # A2
    x = tf.nn.conv2d(x, weights['conv2_1x1_S1'], strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_S1'])  # shape=(?, 128, 128, 64)
    x = tf.nn.conv2d(x, weights['conv2_3x3_S1'], strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_S1'])  # shape=(?, 128, 128, 256)
    x = tf.nn.local_response_normalization(x, depth_radius=5 / 2.0, bias=2.0, alpha=1e-4, beta=0.75)
    C2 = x  # shape=(?, 128, 128, 256)
    x = tf.nn.max_pool(x, ksize=pooling['pool2_3x3_S2'], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 64, 64, 256)

    # inception 3
    inception_3a = inception_unit(inputdata=x, weights=conv_W_3a, biases=conv_B_3a)  # 输入的conv_W_3a是好几个卷积层叠在一起，然后输出的shape是shape=(?, 64, 64, 256)
    inception_3b = inception_unit(inception_3a, weights=conv_W_3b, biases=conv_B_3b)  # shape=(?, 64, 64, 480)
    C3 = conv_1_1(inception_3b, output_channels=512, name='C3')  # shape=(?, 64, 64, 512)

    # 池化层
    x = inception_3b
    x = tf.nn.max_pool(x, ksize=pooling['pool3_3x3_S2'], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 32, 32, 480)

    # inception 4
    inception_4a = inception_unit(inputdata=x, weights=conv_W_4a, biases=conv_B_4a)  # shape=(?, 32, 32, 512)
    # 引出第一条分支
    # softmax0 = inception_4a
    inception_4b = inception_unit(inception_4a, weights=conv_W_4b, biases=conv_B_4b)  # shape=(?, 32, 32, 512)
    inception_4c = inception_unit(inception_4b, weights=conv_W_4c, biases=conv_B_4c)  # shape=(?, 32, 32, 512)
    inception_4d = inception_unit(inception_4c, weights=conv_W_4d, biases=conv_B_4d)  # shape=(?, 32, 32, 528)
    # 引出第二条分支
    # softmax1 = inception_4d
    inception_4e = inception_unit(inception_4d, weights=conv_W_4e, biases=conv_B_4e)  # shape=(?, 32, 32, 832)

    # 池化
    x = inception_4e
    C4 = conv_1_1(x, output_channels=1024, name='C4')  # shape=(?, 32, 32, 1024)
    x = tf.nn.max_pool(x, ksize=pooling['pool4_3x3_S2'], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 16, 16, 832)

    # inception 5
    inception_5a = inception_unit(x, weights=conv_W_5a, biases=conv_B_5a)  # shape=(?, 16, 16, 832)
    inception_5b = inception_unit(inception_5a, weights=conv_W_5b, biases=conv_B_5b)  # shape=(?, 16, 16, 1024)
    softmax2 = inception_5b  # 为什么要叫softmax？？？这儿没用softmax吧？？？？
    C5 = conv_1_1(softmax2, output_channels=2048, name='C5')  # shape=(?, 16, 16, 2048)

    # 后连接
    softmax2 = tf.nn.avg_pool(softmax2, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')
    # softmax2 = tf.nn.dropout(softmax2, keep_prob=keep_prob)  # shape=(?, 16, 16, 1024)
    # 不想dropout了，感觉不但没用而且有时候有害。。
    softmax2 = tf.reshape(softmax2, [-1, weights['FC2'].get_shape().as_list()[0]])  # shape=(?, 200704)
    softmax2 = tf.nn.bias_add(tf.matmul(softmax2, weights['FC2']), biases['FC2'])  # shape=(?, FLAGS.num_classes)
    # print(softmax2.get_shape().as_list())
    return [C1, C2, C3, C4, C5], softmax2