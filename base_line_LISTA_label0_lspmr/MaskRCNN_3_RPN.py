#coding=utf-8
import tensorflow as tf
import MaskRCNN_2_ResNet_and_other_FEN
def build_rpn_model_with_reuse(feature_map, anchor_stride, anchors_per_location, depth):
    assert feature_map.shape[3] == depth
    with tf.variable_scope("rpn_conv_shared"):  # 就仍然用原来的那个name作为这个的variable_scope。。
        shared = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(feature_map, [3, 3, depth, 512], [512],
                   name="rpn_conv_shared", strides=[1, anchor_stride, anchor_stride, 1], padding='SAME')
    with tf.variable_scope("rpn_class_raw"):
        x = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(shared, [1, 1, 512, 2*anchors_per_location], [2 * anchors_per_location],
                   name="rpn_class_raw", strides=[1, 1, 1, 1], padding='VALID')
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")
    with tf.variable_scope("rpn_bbox_pred"):
        x = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(shared, [1, 1, 512, anchors_per_location*4], [anchors_per_location*4],
                   name="rpn_bbox_pred", strides=[1, 1, 1, 1], padding='VALID')
        rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model_with_reuse1(feature_map, anchor_stride, anchors_per_location, depth, config):
    """加一个带最终类别预测的，实现深度监督。
    弄了个深度监督的类别，final_class_logitshe final_probs
    应该暂时不用管那个final_bbox，因为想要监督外接矩形的时候，就选出来相应的rpn_bbox和rpn_bbox_gr似乎就可以了。。"""
    assert feature_map.shape[3] == depth
    with tf.variable_scope("rpn_conv_shared"):  # 就仍然用原来的那个name作为这个的variable_scope。。
        shared = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(feature_map, [3, 3, depth, 512], [512],
                     name="rpn_conv_shared", strides=[1, anchor_stride, anchor_stride, 1], padding='SAME')
    with tf.variable_scope("rpn_class_raw"):
        x = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(shared, [1, 1, 512, 2*anchors_per_location], [2 * anchors_per_location],
                    name="rpn_class_raw", strides=[1, 1, 1, 1], padding='VALID')
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")
    with tf.variable_scope("final_class"):
        x = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(shared, [1, 1, 512, config.NUM_CLASSES * anchors_per_location],
                    [config.NUM_CLASSES * anchors_per_location], name="rpn_bbox_pred", strides=[1, 1, 1, 1], padding='VALID')
        final_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, config.NUM_CLASSES])
        final_probs = tf.nn.softmax(final_class_logits, name="final_class_softmax")
    with tf.variable_scope("rpn_bbox_pred"):
        x = MaskRCNN_2_ResNet_and_other_FEN.conv_layer(shared, [1, 1, 512, anchors_per_location*4], [anchors_per_location*4],
                    name="rpn_bbox_pred", strides=[1, 1, 1, 1], padding='VALID')
        rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])
    return [rpn_class_logits, rpn_probs, rpn_bbox, final_class_logits, final_probs]