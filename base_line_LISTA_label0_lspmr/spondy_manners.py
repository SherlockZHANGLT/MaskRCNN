import tensorflow as tf
import numpy as np
import scipy.misc
from scipy import ndimage
from skimage import transform
#################################################################################################################
#                                             以下是np下数据处理相关的东西                                         #
#################################################################################################################
def clip_and_resize(patient_id, image, box, wanted_shape_x, wanted_shape_y, mode):
    """
    先把原图和掩膜按照掩膜的位置，裁剪为256*256（这个可以调整），然后再缩放回512*512的图。
    wanted_shape_x是输入图像宽度，也希望是裁剪并放缩后的图像宽度。
    """
    assert mode in ['RGB', 'gray']
    if mode == 'RGB':  # shape应该是(512, 512, 3)这样的
        ori_shape_x, ori_shape_y, _ = image.shape
    else:  # shape应该是(512, 512)这样的
        ori_shape_x, ori_shape_y = image.shape
    # if patient_id == 63:
    #     print('检查')
    [y1, x1, y2, x2] = box
    original_length = y2 - y1  # 此处不+1，因为y2世界上已经加了1了，见extract_box函数。
    original_width = x2 - x1
    cliped_size = max(original_length, original_width) + 20  # 裁剪后的区域大小
    # print(cliped_size)
    y_center = np.int32(y1 + original_length / 2)  # 变回int32，让他和原来的y1啥的类型一样。
    x_center = np.int32(x1 + original_width / 2)  #
    selected_area_x1 = np.int32(x_center - cliped_size / 2)  # 从中心点往左数cliped_size/2个点
    selected_area_x2 = np.int32(x_center + cliped_size / 2 - 1)  # 从中心点往右数cliped_size/2 - 1个点
    selected_area_y1 = np.int32(y_center - cliped_size / 2)  # 从中心点往上数cliped_size/2个点
    selected_area_y2 = np.int32(y_center + cliped_size / 2 - 1)  # 从中心点往下数cliped_size/2 - 1个点
    edge_clip = True
    if edge_clip:
        """
        下面那个超出原图边界的东西有点奇怪，原来以为不平移就会报错的，但谁知不平移居然也可以。
        比如说有一次selected_area_y2都到了518了，超过512了，但是
            mask[selected_area_y1:selected_area_y2 + 1, selected_area_x1:selected_area_x2 + 1]
            这句居然没报错。而且验证了一下还是对的。
        不知道为啥啊？

        注意，那个selected_area_y1/x1最好让他>=0,0是可以取的，而selected_area_y2/x2必须<ori_shape_y/x，不能取=号的。
        """
        if selected_area_x1 < 0:
            derta_x = 0 - selected_area_x1
            # print('\r左边界超出原图范围，往右平移。')
            selected_area_x1 += derta_x
            selected_area_x2 += derta_x
        if selected_area_x2 >= ori_shape_x:
            derta_x = selected_area_x2 - ori_shape_x + 1
            # print('\r右边界超出原图范围，往左平移。')
            selected_area_x1 -= derta_x
            selected_area_x2 -= derta_x
        if selected_area_y1 < 0:  # 一般不会，不过还是写上吧。
            derta_y = 0 - selected_area_y1
            # print('\r左边界超出原图范围，往右平移。')
            selected_area_y1 += derta_y
            selected_area_y2 += derta_y
        if selected_area_y2 >= ori_shape_y:
            derta_y = selected_area_y2 - ori_shape_y + 1
            # print('\r右边界超出原图范围，往左平移。')
            selected_area_y1 -= derta_y
            selected_area_y2 -= derta_y
    if mode == 'RGB':
        selected_image = image[selected_area_y1:selected_area_y2 + 1, selected_area_x1:selected_area_x2 + 1, :]
    else:
        selected_image = image[selected_area_y1:selected_area_y2 + 1, selected_area_x1:selected_area_x2 + 1]
    if selected_image.shape[0] != selected_image.shape[1]:
        print(selected_image.shape)  # 有两张图仍然会这样，好像是38和91（还是103来着记不清了），不过都是CT的，先不管了。
    image_selected_and_resized = scipy.misc.imresize(selected_image, (wanted_shape_y, wanted_shape_x), interp='nearest')
    # 奇怪的是，不管RGB还是gray，上句话是不变的，都是(wanted_shape_y, wanted_shape_x)。
    # 但是，很讨厌的是，这个函数居然会改变图像矩阵里每个数的数值，没想到啊。。。所以不得不重新做个灰度放缩。
    # todo 改成transform.resize，见45(get_feature_for_share).py。
    max_before = np.max(image)
    max_after = np.max(image_selected_and_resized)
    image_selected_and_resized = image_selected_and_resized/max_after*max_before  # 让这个图的最大值和输入的一样。
    image_selected_and_resized = image_selected_and_resized.astype(np.uint8)
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(image_selected_and_resized)  # selected_image
    # savename = 'cropped_image' + str(patient_id) + '.png'
    # plt.savefig(savename, bbox_inches="tight", pad_inches=0, ax=8)
    return image_selected_and_resized
def get_useful_mask(mask_input, box_confirmed):
    mask_pros = np.zeros_like(mask_input)
    for i in range(box_confirmed.shape[0]):
        y1 = box_confirmed[i, 0]
        y2 = box_confirmed[i, 2]
        x1 = box_confirmed[i, 1]
        x2 = box_confirmed[i, 3]
        mask_pros[y1:y2, x1:x2] = 1
    mask_output = mask_input * mask_pros
    return mask_output
def get_batch_inputs_for_spondy(dataset, image_ids):
    """和MRCNN的那个类似，但是应该简单多了，因为我现在就直接从dataset中弄出来图像、掩膜、标签等就可以了。"""
    batch_images = []
    batch_gt_mask = []
    batch_labels = []
    for image_id in image_ids:
        image = dataset.image_info[image_id]['image']
        mask = dataset.image_info[image_id]['mask']
        label = dataset.image_info[image_id]['label']
        batch_images.append(image)
        batch_gt_mask.append(mask)
        batch_labels.append(label)
    batch_images = np.stack(batch_images, axis=0)  # (batch_size, 224, 224, 3)
    batch_gt_mask = np.stack(batch_gt_mask, axis=0)  #  应该是(batch_size, 224, 224)
    batch_labels = np.stack(batch_labels, axis=0)  # (batch_size,)
    inputs_for_spondy_dict_feeding = [batch_images, batch_gt_mask, batch_labels]
    return inputs_for_spondy_dict_feeding
def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size, maybe mirror horizontally
    输入应该是一张RGB图像。
    """
    init_shape = image.shape
    assert len(init_shape)==2 or len(init_shape)==3  # 要么是灰度图，要么是RGB图。
    if len(init_shape)==3:
        new_shape = [int(init_shape[0] + pad * 2),
                     int(init_shape[1] + pad * 2),
                     init_shape[2]]
        zeros_padded = np.zeros(new_shape, dtype=image.dtype)
        zeros_padded[pad:int(init_shape[0] + pad), pad:int(init_shape[1] + pad), :] = image
        # zeros_padded = np.pad(image, ((pad,pad), (pad,pad), (0,0)), mode='symmetric')
        """以上，补零"""
        # randomly crop to original size
        init_x = np.random.randint(0, int(pad * 2))
        init_y = np.random.randint(0, int(pad * 2))
        cropped = zeros_padded[
            init_x: int(init_x + init_shape[0]),
            init_y: int(init_y + init_shape[1]),
            :]  # 这个cropped的形状和输入图像的形状一样。
        """以上，随机裁剪。注意是在补零后的图像上做的。"""
        # rotate
        random_angle = np.random.randint(-10, 10)
        # print ('旋转角度是：%d' % random_angle)
        # rotated = rotate(cropped[:,:,0], random_angle)  # 这个对吗？只转了3个通道里的1个，那么省下两个通道没有转啊。。
        h, w, _ = cropped.shape
        img_rote1 = ndimage.rotate(cropped, random_angle)  # 换一种方法来试试。。所有通道都旋转了。。
        h1, w1, _ = img_rote1.shape
        center_x = int(h1 / 2)
        center_y = int(w1 / 2)
        left = int(center_x - h / 2)
        right = int(center_x + h / 2)
        up = int(center_y - w / 2)
        down = int(center_y + w / 2)
        rotated = img_rote1[left:right, up:down, :]  #
        """以上，随机旋转一个小角度，-10~10°。"""
        cropped = rotated  # 改了的
    else:
        new_shape = [int(init_shape[0] + pad * 2),
                     int(init_shape[1] + pad * 2)]
        zeros_padded = np.zeros(new_shape, dtype=image.dtype)
        zeros_padded[pad:int(init_shape[0] + pad), pad:int(init_shape[1] + pad)] = image
        """以上，补零"""
        # randomly crop to original size
        init_x = np.random.randint(0, int(pad * 2))
        init_y = np.random.randint(0, int(pad * 2))
        cropped = zeros_padded[
                  init_x: int(init_x + init_shape[0]),
                  init_y: int(init_y + init_shape[1])]  # 这个cropped的形状和输入图像的形状一样。
        """以上，随机裁剪。注意是在补零后的图像上做的。"""
        # rotate
        random_angle = np.random.randint(-10, 10)
        # print('旋转角度是：%d' % random_angle)
        h, w = cropped.shape
        img_rote1 = ndimage.rotate(cropped, random_angle)  # 用我的那个函数吧
        h1, w1 = img_rote1.shape
        center_x = int(h1 / 2)
        center_y = int(w1 / 2)
        left = int(center_x - h / 2)
        right = int(center_x + h / 2)
        up = int(center_y - w / 2)
        down = int(center_y + w / 2)
        rotated = img_rote1[left:right, up:down]  #
        """以上，随机旋转一个小角度，-10~10°。"""
        cropped = rotated
    return cropped
def grayscals_image_norm(image, RGB_or_gray):
    assert RGB_or_gray in ['RGB', 'gray']
    if RGB_or_gray== 'gray':
        """输入的image应该是shape=(batch_size, h, w)的"""
        image = image.astype('float32')
        image_normed = (image - np.mean(image)) / np.std(image)
    else:
        image_normed = image  # 一开始让他和image一样。
        for i in range(3):
            image[:,:,:,i]=image[:,:,:,i].astype('float32')
            image_normed[:,:,:,i] = (image[:,:,:,i] - np.mean(image[:,:,:,i])) / np.std(image[:,:,:,i])
    return image_normed
def make_dense_label_to_1_hot_label(dense_label, num_classes):
    dense_label = dense_label.astype(int)  # 1、得先变成int才行，2、矩阵要这样转换类型。。
    dense_label = np.asarray(dense_label)
    num_labels = dense_label.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + dense_label.ravel()] = 1
    return labels_one_hot
def chesk_func(labels_feed, patiend_id_used_in_this_batch, predicted_logits, show_all, show_wrong):
    # 检查哪个病人分类错误。
    c = 0
    for j in range(len(patiend_id_used_in_this_batch)):
        label = labels_feed[j]
        patiend_id = patiend_id_used_in_this_batch[j]
        _logit = predicted_logits[j, :]
        l = _logit.tolist()
        if show_all:  # 是否都打印出来
            print('病人序号：', patiend_id, '金标准类别：', label)
            print('判断出来的logits：', _logit, '判断的类别：', l.index(max(l)))
        if show_wrong and l.index(max(l)) != label:  # 判断错了的要打印出来
            print('搞错的病人序号：', patiend_id, '金标准类别：', label, '判断的类别：', l.index(max(l)))
        if label == l.index(max(l)):
            c=c+1
    print("在这个批次（一共%d个）中，预测对了%d个。" % (len(patiend_id_used_in_this_batch), c))
    return c
#################################################################################################################
#                                             以下是网络构建相关的东西                                             #
#################################################################################################################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    w = tf.Variable(initial)
    return w
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    b = tf.Variable(initial)
    return b
# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)
def batch_norm(input):
    normed = tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None)
    # normed = tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None,
    #                                       renorm=True, renorm_clipping=dict(rmax=3, rmin=0, dmax=5), renorm_decay=0.99)  这个不行。。。
    return normed
def group_norm(input, G=32, eps=1e-5, scope='group_norm'):
    # normed = tf.contrib.layers.group_norm(input, groups=32, channels_axis=-1, reduction_axes=(-3, -2), center=True,
    #                                       scale=True, epsilon=1e-3)  这个好像不行，可能是tf版本太低吧。
    with tf.variable_scope(scope):
        _, H, W, C = input.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(input, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
        normed = tf.reshape(x, [-1, H, W, C]) * gamma + beta
    return normed
def train_MaskMRCNN_model(BATCH_SIZE, num_epochs_per_decay, learning_rate_decay_type, LEARNING_RATE, end_learning_rate,
                          learning_rate_decay_factor, LEARNING_MOMENTUM,
                          vars, num_samples_per_epoch, loss, spondy_global_step):
    # 训练上面的模型，得到train_op。原来是把二者合一的，感觉不太好就拆开了。
    """【基础】全局步数
    global_step必须放在这里面，才能起到作用。。
    """
    decay_steps = int(num_samples_per_epoch / BATCH_SIZE * num_epochs_per_decay)
    # 上句，通过1个时代中的样本数，还有1个批次中的样本数，就能算出1个时代有多少个批次，然后再除以多少时代衰减1次学习率，就能知道过多少步衰减一次学习率。
    if learning_rate_decay_type == 'exponential':
        configured_learning_rate = tf.train.exponential_decay(LEARNING_RATE, spondy_global_step, decay_steps,
                                                              learning_rate_decay_factor, staircase=True,
                                                              name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        configured_learning_rate = tf.constant(LEARNING_RATE, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        configured_learning_rate = tf.train.polynomial_decay(LEARNING_RATE, spondy_global_step, decay_steps,
                                                             end_learning_rate, power=1.0, cycle=False,
                                                             name='polynomial_decay_learning_rate')
    else:
        raise ValueError('无法识别这样的学习率衰减类别(learning_rate_decay_type) [%s]',
                         learning_rate_decay_type)
    optimizer = tf.train.MomentumOptimizer(configured_learning_rate, momentum=LEARNING_MOMENTUM, name='Momentum')
    grads = optimizer.compute_gradients(loss, var_list=vars)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)
    train_op = optimizer.apply_gradients(grads, global_step=spondy_global_step)  # 此处全局步数自增。
    return train_op, configured_learning_rate
def train_and_inference(image_pl, label_pl, keep_pl, FLAGS, num_samples_per_epoch, train_flag):
    """【基础】占位符不是在这个计算类别（logits）、损失的函数（其实就是所谓的“网络”）里定义的，而是在外面定义好了传进来的！"""
    _, h, w, _ = image_pl.shape  # image_pl的shape应该是[batch_size, 512, 512, 3]
    with tf.variable_scope("spondy_grading"):
        W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_1 = bias_variable([64])
        conv1_1 = conv2d(image_pl, W_conv1_1) + b_conv1_1
        # output = tf.nn.relu(batch_norm(conv1_1))  # 这儿他是每次都用了批次正则化的啊。。shape=(?, 512, 512, 64)。
        output = tf.nn.relu(group_norm(conv1_1, G=32, eps=1e-5, scope='conv1_1'))  # 改用分组正则化试试。shape=(?, 512, 512, 64)。
        # 原来用cifar是shape=(?, 32, 32, 64)，dtype都是float32
        W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_2 = bias_variable([64])
        conv1_2 = conv2d(output, W_conv1_2) + b_conv1_2
        # output = tf.nn.relu(batch_norm(conv1_2))
        output = tf.nn.relu(group_norm(conv1_2, G=32, eps=1e-5, scope='conv1_2'))
        output = max_pool(output, 2, 2, "pool1")  # shape=(?, 256, 256, 64)。原来用cifar是shape=(?, 16, 16, 64)
        h_pool1 = output
        W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_1 = bias_variable([128])
        conv2_1 = conv2d(output, W_conv2_1) + b_conv2_1
        # output = tf.nn.relu(batch_norm(conv2_1))  # shape=(?, 256, 256, 128)。原来用cifar是shape=(?, 16, 16, 128)
        output = tf.nn.relu(group_norm(conv2_1, G=32, eps=1e-5, scope='conv2_1'))  #
        W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_2 = bias_variable([128])
        conv2_2 = conv2d(output, W_conv2_2) + b_conv2_2
        # output = tf.nn.relu(batch_norm(conv2_2))
        output = tf.nn.relu(group_norm(conv2_2, G=32, eps=1e-5, scope='conv2_2'))
        output = max_pool(output, 2, 2, "pool2")  # shape=(?, 128, 128, 128)。原来用cifar是shape=(?, 8, 8, 128)
        h_pool2 = output
        W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_1 = bias_variable([256])
        conv3_1 = conv2d(output, W_conv3_1) + b_conv3_1
        # output = tf.nn.relu(batch_norm(conv3_1))  # shape=(?, 128, 128, 256)。原来用cifar是shape=(?, 8, 8, 256)
        output = tf.nn.relu(group_norm(conv3_1, G=32, eps=1e-5, scope='conv3_1'))  #
        W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_2 = bias_variable([256])
        conv3_2 = conv2d(output, W_conv3_2) + b_conv3_2
        # output = tf.nn.relu(batch_norm(conv3_2))  # shape=(?, 128, 128, 256)。原来用cifar是shape=(?, 8, 8, 256)
        output = tf.nn.relu(group_norm(conv3_2, G=32, eps=1e-5, scope='conv3_2'))  #
        W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_3 = bias_variable([256])
        conv3_3 = conv2d(output, W_conv3_3) + b_conv3_3
        # output = tf.nn.relu(batch_norm(conv3_3))  # shape=(?, 128, 128, 256)。原来用cifar是shape=(?, 8, 8, 256)
        output = tf.nn.relu(group_norm(conv3_3, G=32, eps=1e-5, scope='conv3_3'))  #
        W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_4 = bias_variable([256])
        conv3_4 = conv2d(output, W_conv3_4) + b_conv3_4
        # output = tf.nn.relu(batch_norm(conv3_4))
        output = tf.nn.relu(group_norm(conv3_4, G=32, eps=1e-5, scope='conv3_4'))
        output = max_pool(output, 2, 2, "pool3")  # shape=(?, 64, 64, 256)。原来用cifar是shape=(?, 4, 4, 256)
        h_pool3 = output
        W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_1 = bias_variable([512])
        conv4_1 = conv2d(output, W_conv4_1) + b_conv4_1
        # output = tf.nn.relu(batch_norm(conv4_1))  # shape=(?, 64, 64, 512)。原来用cifar是shape=(?, 4, 4, 512)
        output = tf.nn.relu(group_norm(conv4_1, G=32, eps=1e-5, scope='conv4_1'))  #
        W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_2 = bias_variable([512])
        conv4_2 = conv2d(output, W_conv4_2) + b_conv4_2
        # output = tf.nn.relu(batch_norm(conv4_2))  # shape=(?, 64, 64, 512)。原来用cifar是shape=(?, 4, 4, 512)
        output = tf.nn.relu(group_norm(conv4_2, G=32, eps=1e-5, scope='conv4_2'))  #
        W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_3 = bias_variable([512])
        conv4_3 = conv2d(output, W_conv4_3) + b_conv4_3
        # output = tf.nn.relu(batch_norm(conv4_3))  # shape=(?, 64, 64, 512)。原来用cifar是shape=(?, 4, 4, 512)
        output = tf.nn.relu(group_norm(conv4_3, G=32, eps=1e-5, scope='conv4_3'))  #
        W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_4 = bias_variable([512])
        conv4_4 = conv2d(output, W_conv4_4) + b_conv4_4
        # output = tf.nn.relu(batch_norm(conv4_4))
        output = tf.nn.relu(group_norm(conv4_4, G=32, eps=1e-5, scope='conv4_4'))
        output = max_pool(output, 2, 2)  # shape=(?, 32, 32, 512)。原来用cifar是shape=(?, 2, 2, 512)
        h_pool4 = output
        W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_1 = bias_variable([512])
        conv5_1 = conv2d(output, W_conv5_1) + b_conv5_1
        # output = tf.nn.relu(batch_norm(conv5_1))  # shape=(?, 32, 32, 512)。原来用cifar是shape=(?, 2, 2, 512)
        output = tf.nn.relu(group_norm(conv5_1, G=32, eps=1e-5, scope='conv5_1'))  #
        W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_2 = bias_variable([512])
        conv5_2 = conv2d(output, W_conv5_2) + b_conv5_2
        # output = tf.nn.relu(batch_norm(conv5_2))  # shape=(?, 32, 32, 512)。原来用cifar是shape=(?, 2, 2, 512)
        output = tf.nn.relu(group_norm(conv5_2, G=32, eps=1e-5, scope='conv5_2'))  #
        W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_3 = bias_variable([512])
        conv5_3 = conv2d(output, W_conv5_3) + b_conv5_3
        # output = tf.nn.relu(batch_norm(conv5_3))  # shape=(?, 32, 32, 512)。原来用cifar是shape=(?, 2, 2, 512)
        output = tf.nn.relu(group_norm(conv5_3, G=32, eps=1e-5, scope='conv5_3'))  #
        W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_4 = bias_variable([512])
        conv5_4 = conv2d(output, W_conv5_4) + b_conv5_4
        # output = tf.nn.relu(batch_norm(conv5_4))  # shape=(?, 32, 32, 512)。原来用cifar是shape=(?, 2, 2, 512)
        output = tf.nn.relu(group_norm(conv5_4, G=32, eps=1e-5, scope='conv5_4'))  #
        output = max_pool(output, 2, 2, "pool5")  # 原来没有，池化了之后是shape=(?, 16, 16, 512)
        h_pool5 = output
        # 再加一个卷积层+池化层，避免OOM的
        W_conv6 = tf.get_variable('conv6', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv6 = bias_variable([512])
        conv6 = conv2d(output, W_conv6) + b_conv6
        # output = tf.nn.relu(batch_norm(conv6))  # shape=(?, 16, 16, 512)
        output = tf.nn.relu(group_norm(conv6, G=32, eps=1e-5, scope='conv6'))  #
        output = max_pool(output, 4, 4, "pool6")  # shape=(?, 8, 8, 512)
        h_pool6 = output
        _, a, b, c = output.get_shape().as_list()
        output = tf.reshape(output, [-1, a * b * c])  # 原来用cifar是shape=(?, 2048)
        W_fc1 = tf.get_variable('fc1', shape=[a * b * c, 4096], initializer=tf.contrib.keras.initializers.he_normal())  # 原来都是4096，一共6处都改成1024
        #### 感觉就是这个fc1层有问题，但是一时也不知道为什么啊。。。
        b_fc1 = bias_variable([4096])
        output = tf.matmul(output, W_fc1) + b_fc1
        # 全连接层后面确实就不能加bn了，全连接后面没用bn的情况，似乎测试的时候batch_size是不影响的。。
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, keep_pl)  # 原来用cifar是shape=(?, 4096)
        h_fc1 = output
        W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc2 = bias_variable([4096])
        output = tf.matmul(output, W_fc2) + b_fc2
        # output = batch_norm(output)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, keep_pl)  # 原来用cifar是shape=(?, 4096)
        h_fc2 = output
        W_fc3 = tf.get_variable('fc3', shape=[4096, FLAGS.num_classes], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc3 = bias_variable([FLAGS.num_classes])
        # logits = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))  # 原来用cifar是shape=(?, 10)。。。这个ReLU有问题啊。。。
        output = tf.matmul(output, W_fc3) + b_fc3
        # logits = batch_norm(output)  # 原来用cifar是shape=(?, 10)。。。这个ReLU有问题啊。。。
        logits = output  # 原来用cifar是shape=(?, 10)。。。这个ReLU有问题啊。。。
    label_backup = label_pl
    batch_size_here = tf.size(label_pl)
    label_pl = tf.expand_dims(label_pl, 1)
    indices = tf.expand_dims(tf.range(0, batch_size_here), 1)
    concated = tf.concat([indices, label_pl], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size_here, FLAGS.num_classes]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    """以上，计算预测值，并且用交叉熵计算损失函数。以下，开始弄训练的方法。"""
    # vars = tf.trainable_variables()  # 似乎有问题啊，这儿应该是只训练VGG的这些啊。。。
    all_vars = tf.all_variables()
    vars = [var for var in all_vars if 'spondy_grading' in var.name]  # 【基础】这个vars是只有VGG的那些变量。。
    l2_weight = FLAGS.l2_weight
    # l2_loss = (tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])) * l2_weight
    l2_loss = (tf.add_n([tf.nn.l2_loss(var) for var in vars])) * l2_weight  # 这个也得改成只在滑脱分级网络里找变量。
    loss_with_l2 = loss + l2_loss
    """以上，先搞出来所有可训练的变量，加入L2正则化项。
    【基础】L2正则化
    1、计算，就是所有变量平方和再开方，然后乘以权重系数。有可能还要除以变量个数，即tf.size(var)（如果是3*3矩阵，那么这个size就是9），
        这就是一个变量（比如说一个卷积核）的L2损失，把它加到collection里，然后求和，就是总的L2损失。
    2、权重选取，那个权重系数据说应该是1e-4左右。
    """

    correct = tf.nn.in_top_k(logits, label_backup, 1)
    # correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_backup, 1))
    mean_correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))
    # mean_correct_num = tf.reduce_mean(tf.cast(correct, tf.float32))
    """以上，计算每个批次中，弄对的图像张数。mean_correct_num应该是这个批次的batch_size张图里，搞对的图像张数。
    【基础】注意是用reduce_sum还是reduce_mean。
    这取决于主函数里，想要run掉的是这个批次预测对的总数，还是平均正确率。
        那个correct应该是一堆0-1向量，向量长度是batch_size，其中每个数如果是1就表示这张图对了，否则错了。
        所以说，如果用reduce_sum，就得到正确的张数；如果是reduce_mean，就是正确率。
        个人而言，更习惯这儿用总数，然后在外面都加起来，最后除以数据集中的数据个数，就得到正确率。
    """
    others_for_check = [h_pool1, h_pool2, h_pool3, h_pool4, h_pool5, h_pool6, h_fc1, h_fc2, label_backup, onehot_labels, correct]  #
    return logits, loss, l2_loss, loss_with_l2, mean_correct_num, others_for_check
#################################################################################################################
#                                        以下是想复用resnet特征用的东西                                            #
#################################################################################################################
def get_resized_feature_for_one_image(P2, P3, P4, P5, roi_level, roi_norm, wanted_shape):
    """
    输入的feature应该是[P2, P3, P4, P5]，每一个P*都是(h_this_level, w_this_level, c_this_level)的张量，即：
        这些P*都是张量，而且不同的张量之间，宽高和通道数都是不一样的。
    先用roi_level选择一个特征，然后用roi_norm*特征宽度裁剪，然后放缩到(wanted_shape, wanted_shape, c_this_level)。
    """
    assert roi_level in [2, 3, 4, 5], 'roi_level必须是2~5中的一个。'
    # print('roi_level=', roi_level, '\n')
    if roi_level == 2:
        feature_selected = P2
        feature_size = feature_selected.shape[0]
    elif roi_level == 3:
        feature_selected = P3
        feature_size = feature_selected.shape[0]
    elif roi_level == 4:
        feature_selected = P4
        feature_size = feature_selected.shape[0]
    else:
        feature_selected = P5
        feature_size = feature_selected.shape[0]
    [y1, x1, y2, x2] = np.around(roi_norm * feature_size).astype(np.int32)  # 先四舍五入，再变成int类型。
    # [y1, x1, y2, x2] = np.around(roi_norm * feature_size).astype(np.int32)[0]  # 先四舍五入，再变成int类型。
    """【大坑】
    如果用带[0]的那个做，网络是没错，但是run的时候会出错说什么“Invalid argument: TypeError: 'numpy.int32' object is not iterable”。
    但，关键是我都不知道怎么就iterable了，超级烦啊。。。
    其实后来发现，问题是出在单独拎出来这个程序跑的时候，输入的那个roi_norm的shape是(1,4)，
        而如果把这个东西融合到总的程序里的话，这个roi_norm的shape就是(4,)了，所以才会有这个问题。
    【检查】
    弄到tf程序里的py_func不太好查，目前只能采取在每一步之前或者之后都print出来点东西的方法。然后实际在构建网络的时候，
        所有东西都不会打印出来，只有在run的时候才会打印，而且一般都是一次就打印出来batch_size个东西。。
    """
    feature_cropped = feature_selected[y1:y2 + 1, x1:x2 + 1, :]
    feature_resized = transform.resize(feature_cropped, wanted_shape, preserve_range=True)
    # 上面这个transform.resize基本上保持了图像像素值不变的。看起来比那个scipy.misc.imresize好一些。
    # 基础，目标形状只需要写2维就可以了，通道数会自动保存成和原来通道数一样的。
    feature_resized = feature_resized.astype(np.float32)
    return feature_resized

import MaskRCNN_2_ResNet
import MaskRCNN_6_heads
def grading_network(wanted_shape, wanted_shape_feature, ori_image_area, mrcnn_feature_maps, input_images,
                    input_labels_pl_grading, result_rois_refined_batch, config, config_sp, grading_classes):
    """就是做滑脱分级的函数。"""
    # result_rois_batch.set_shape([None, 4])  # 确认过，这个result_rois_batch应该就是(?(批大小), 4)。
    [P2, P3, P4, P5] = mrcnn_feature_maps  # 是个list，里面有4个张量，shape=(2, 128, 128, 256)、(2, 64, 64, 256)、(2, 32, 32, 256)、(2, 16, 16, 256)
    _, _, _, channel_num = P2.get_shape().as_list()
    selected_images_resized = []  # 不知道有没有用，先留着
    selected_features_resized = []  # 特征
    for i in range(config.BATCH_SIZE):
        input_image_this = input_images[i, :, :, :]
        # result_rois_norm = result_rois_batch[i, :]  # 归一化的尺度
        result_rois_norm = result_rois_refined_batch[i, :]  # 归一化的尺度
        result_rois = tf.cast(result_rois_norm * 512,
                              tf.int32)  # result_rois_batch/result_rois_refined_batch是归一化了的，这儿变回成没归一化的。。
        selected_image = input_image_this[result_rois[0]:result_rois[2] + 1, result_rois[1]:result_rois[3] + 1, :]
        selected_image_resized = tf.image.resize_images(selected_image, [wanted_shape, wanted_shape], method=0)
        # 上句是这一张图先裁剪再放缩。输出应该是shape=(wanted_shape, wanted_shape, 3)
        selected_images_resized.append(selected_image_resized)
        """以上是先把图给裁剪缩放了，不知道有没有用，先放这儿。"""
        P2_this = P2[i, :, :, :]  # shape=(128, 128, 256)
        P3_this = P3[i, :, :, :]  # (64, 64, 256)
        P4_this = P4[i, :, :, :]  # (32, 32, 256)
        P5_this = P5[i, :, :, :]  # (16, 16, 256)
        h_normed = (result_rois_norm[2] - result_rois_norm[0])  # 这个是已经归一化了的。shape现在是unknown，run掉后会是(2,)。
        w_normed = (result_rois_norm[3] - result_rois_norm[1])
        roi_level = MaskRCNN_6_heads.log2_graph(tf.sqrt(h_normed * w_normed) / (224.0 / tf.sqrt(ori_image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))  # 这个是一个数 ，
        # 执行完后，是<tf.Tensor 'Minimum:0' shape=() dtype=int32>，
        # 循环一次后，下一张图的roi_level就变成了<tf.Tensor 'Minimum_1:0' shape=() dtype=int32>。可以run掉观察这个批次里的图的roi_level分别是多少。
        feature_resized = tf.py_func(get_resized_feature_for_one_image,
                                     [P2_this, P3_this, P4_this, P5_this, roi_level, result_rois_norm,
                                      wanted_shape_feature],
                                     tf.float32)
        feature_resized.set_shape([wanted_shape_feature[0], wanted_shape_feature[1], channel_num])  # 输出的shape应该是这个。
        selected_features_resized.append(feature_resized)
        """以上，裁剪缩放后的特征。"""
    selected_images_resized_batch = tf.stack(selected_images_resized, axis=0)
    selected_features_resized_batch = tf.stack(selected_features_resized, axis=0)
    # 上句，输出的shape是(2, 64, 64, 256)/(2, 7, 7, 256)，具体见下面注释。
    """以上，得到裁剪、缩放后的图像和特征。下面试着复用特征，构建滑脱分级网络。
    实验发现，如果wanted_shape_feature = (64, 64)，即上面的selected_features_resized_batch的shape是(2, 64, 64, 256)，
        那么，会发现大概在500步左右，训练精度就能达到0.98，但是，正好在OOM的边缘。
    如果wanted_shape_feature = (7, 7)，即上面的selected_features_resized_batch的shape是(2, 7, 7, 256)，
        那么，不会OOM，但是，精度就不怎么高。
    如果wanted_shape_feature = (32, 32)，即上面的selected_features_resized_batch的shape是(2, 32, 32, 256)，
        那么，会发现不会OOM，而且精度不比64, 64的低多少。
    """
    gradings = config_sp.num_gradings  # 0~3这4个分级。
    with tf.variable_scope("spondy_grading"):
        W_conv1 = tf.get_variable('conv1', shape=[3, 3, channel_num, 512],
                                  initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1 = bias_variable([512])
        conv1 = conv2d(selected_features_resized_batch, W_conv1) + b_conv1  # (2, 64, 64, 512)
        GN_relu_1 = tf.nn.relu(group_norm(conv1, G=32, eps=1e-5, scope='conv1'))
        pooled1 = max_pool(GN_relu_1, 2, 2, "pool1")  # shape=(2, 32, 32, 512)
        W_conv2 = tf.get_variable('conv2', shape=[3, 3, 512, 512],
                                  initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2 = bias_variable([512])
        conv2 = conv2d(pooled1, W_conv2) + b_conv2
        GN_relu_2 = tf.nn.relu(group_norm(conv2, G=32, eps=1e-5, scope='conv2'))
        pooled2 = max_pool(GN_relu_2, 2, 2, "pool2")  # shape=应该是(2, 16, 16, 512)
        W_conv3 = tf.get_variable('conv3', shape=[3, 3, 512, 1024],
                                  initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3 = bias_variable([1024])
        conv3 = conv2d(pooled2, W_conv3) + b_conv3
        GN_relu_3 = tf.nn.relu(group_norm(conv3, G=32, eps=1e-5, scope='conv3'))
        pooled3 = max_pool(GN_relu_3, 2, 2, "pool3")  # shape=应该是(2, 8, 8, 1024)
        # x = MaskRCNN_2_ResNet.conv_layer(selected_features_resized_batch,
        #                                  [wanted_shape_feature[0], wanted_shape_feature[1], channel_num, 1024], [1024],
        #                                  strides=[1, 1, 1, 1], padding='VALID', name="grading_1")
        _, h_pooled3, w_pooled3, c_pooled3 = pooled3.get_shape().as_list()
        x = MaskRCNN_2_ResNet.conv_layer(pooled3, [h_pooled3, w_pooled3, c_pooled3, 1024], [1024],
                                         strides=[1, 1, 1, 1], padding='VALID',
                                         name="grading_1")  # shape=(2, 1, 1, 1024)
        x = tf.nn.relu(group_norm(x, G=32, eps=1e-5, scope='grading_bn1'))
        x = MaskRCNN_2_ResNet.conv_layer(x, [1, 1, 1024, 1024], [1024],
                                         strides=[1, 1, 1, 1], padding='VALID',
                                         name="grading_2")  # shape=(2, 1, 1, 1024)
        x = tf.nn.relu(group_norm(x, G=32, eps=1e-5, scope='grading_bn2'))
        shared = tf.squeeze(tf.squeeze(x, axis=1), axis=1, name="grading_squeeze")  # shape=(2, 1024)
        # 基本上和FPN里的一样，只不过这儿没用BN试试了。
        w_grading = MaskRCNN_2_ResNet.weight_variable([1024, gradings])
        b_grading = MaskRCNN_2_ResNet.bias_variable([gradings])
        logits_grading = tf.matmul(shared, w_grading) + b_grading  # shape是(?, 4)。
    is_traing_pl_grading = tf.placeholder(tf.bool)  # 是否训练模式的占位符..........这个不就是is_training吗？？？？？到时候处理掉。。。
    label_pl = tf.expand_dims(input_labels_pl_grading, 1)
    indices = tf.expand_dims(tf.range(0, config.BATCH_SIZE), 1)
    concated = tf.concat([indices, label_pl], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([config.BATCH_SIZE, grading_classes]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_grading, labels=onehot_labels,
                                                            name='xentropy')
    # loss_grading = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    loss_grading = tf.reduce_sum(cross_entropy, name='xentropy_sum') / config.BATCH_SIZE  #
    # 不知道为什么，用reduce_mean会给出不对的结果，明明看到cross_entropy是两个很接近于0的值，结果reduce_mean之后居然变成20多。。。不知道怎么回事。。。
    correct = tf.nn.in_top_k(logits_grading, input_labels_pl_grading, 1)
    mean_correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))
    return logits_grading, loss_grading, mean_correct_num