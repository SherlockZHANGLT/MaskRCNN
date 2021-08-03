#coding=utf-8
import tensorflow as tf
import MaskRCNN_2_ResNet  # 就是想试试这个东西怎么把w和b都弄出来一大堆的nan？！
input = tf.Variable(tf.random_normal([1,64,64,3]))  # 假装是一张图  <tf.Variable 'Variable:0' shape=(1, 64, 64, 3) dtype=float32_ref>
paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
x = tf.pad(input, paddings, "CONSTANT")  # <tf.Tensor 'Pad:0' shape=(1, 70, 70, 3) dtype=float32>
w = MaskRCNN_2_ResNet.weight_variable([7, 7, 3, 64])  # <tf.Variable 'Variable_1:0' shape=(7, 7, 3, 64) dtype=float32_ref>
b = MaskRCNN_2_ResNet.bias_variable([64])  # <tf.Variable 'Variable_2:0' shape=(64,) dtype=float32_ref>
x = tf.nn.conv2d(x, w, strides = [1, 2, 2, 1], padding = 'VALID', name = 'conv_1') + b  # <tf.Tensor 'add:0' shape=(1, 32, 32, 64) dtype=float32>

sess=tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

_x, _w, _b, _input = sess.run([x, w, b, input])
print('x')
print(_x)
print('w')
print(_w)
print('b')
print(_b)
print('input')
print(_input)