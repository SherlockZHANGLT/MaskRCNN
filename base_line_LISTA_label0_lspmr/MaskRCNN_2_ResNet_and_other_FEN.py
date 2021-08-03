#coding=utf-8
import tensorflow as tf
def weight_variable(shape):
    # weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer())
    weights = tf.get_variable("weights", shape, initializer=tf.contrib.keras.initializers.he_normal())
    # 注意这儿用的是tf.get_variable，对比不共享的时候用的是tf.Variable！！！！！
    # 【基础】这儿用了he_normal，从头开始训练就不出nan了！
    return weights
def bias_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.0))
    # 注意这儿用的是tf.get_variable
    return biases
def conv_layer(x, w, b, name, strides, padding = 'SAME'):
    """
    卷积层。其中，有权重w和偏置b两种参数。
    输入是：被卷积的张量x、卷积核w、偏置b；
    输出是：卷积并且加了偏置之后的值，也是一个张量。
    """
    with tf.variable_scope(name):
        w = weight_variable(w)
        b = bias_variable(b)
        conv_and_biased = tf.nn.conv2d(x, w, strides = strides, padding = padding, name = name) + b
    return conv_and_biased
def batch_normalization(inputs, scope, is_training=True, need_relu=True):
    """
    批量正则化，也可以用封装好了的函数。
    输入的inputs是一个张量（比如说，那个卷积函数conv_layer的输出），
        scope应该是那个网络的名字，
        is_training很有可能就是是否乱序的那个控制flag。
    最后，在正则化后又做了一次ReLU。
    输出的就是ReLU后的值。
    """
    bn = tf.contrib.layers.batch_norm(inputs,
        decay=0.999,
        center=True,
        scale=True,  # 原来这个是FALSE，所以这儿输出来的变量没有那个gamma。为了和keras的一样，就改成了True。。
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        param_regularizers=None,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=is_training,  # 如果是true，就accumulate the statistics of the moments into moving_mean and moving_variance，累积统计值。
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        fused=False,
        data_format='NHWC',
        zero_debias_moving_mean=False,
        scope=scope,
        renorm=False,
        renorm_clipping=None,
        renorm_decay=0.99)
    if need_relu:
        bn = tf.nn.relu(bn, name='relu')
    return bn
def group_norm(input, G=32, eps=1e-5, scope='group_norm', need_relu=True):
    # normed = tf.contrib.layers.group_norm(input, groups=32, channels_axis=-1, reduction_axes=(-3, -2), center=True,
    #                                       scale=True, epsilon=1e-3)  这个好像不行，可能是tf版本太低吧。
    with tf.variable_scope(scope):
        _, H, W, C = input.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(input, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))  # BN的时候，这个的shape是[C]，但是感觉GN的时候这么玩好像不行。
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
        normed = tf.reshape(x, [-1, H, W, C]) * gamma + beta
        if need_relu:
            normed = tf.nn.relu(normed, name='relu')
    return normed
def maxpooling(x,kernal_size, strides, name):  # 最大池化，前面的是核大小，一般为[1, 2, 2, 1]，后面的strides指的是步长，如[1, 2, 2, 1]。
    return tf.nn.max_pool(x, ksize=kernal_size, strides=strides, padding='SAME',name = name)
def avg_pool(input_feats, k):
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(input_feats, ksize, strides, padding)  # 平均池化
    return output
def deconv_layer(x,w,output_shape,name=None):  # 这个先不用了！！！！！！
    """反卷积层，和卷积的实现方法差不多。只不过这儿的strides是固定的，然后也没有那个偏置项b啊。
    那个data_format是说，输出的四维分别是批次中图像张数N、图像长H、宽W、通道数C。
    不过因为要求输入的那个output_shape实在是太sb了。。
    """
    w = weight_variable(w)  # w是卷积核的[长、宽、输入层数、输出层数]
    return tf.nn.conv2d_transpose(x,w,output_shape,strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC',name=name)
def deconv_layer1(x, out_layer, w_shape, strides=2, padding='SAME', name=None):  # 这个其实也没用到。。
    """这个比起上面那个，好的地方是不用那个output_shape，但坏处是，似乎也没法共享啊。。"""
    y = tf.layers.conv2d_transpose(x, out_layer, w_shape, strides=strides, padding=padding, name=name)
    return y
def deconv_layer_with_reuse(x, w_shape, b_shape, output_shape, strides, padding = 'SAME'):  # 这个反卷积层和不共享的一样，只不过w是通过上面函数弄过来的。
    # 函数是在MaskRCNN_6_heads.py才用到，这个py文件里没用到。
    w = weight_variable(w_shape)  # 输入的w是卷积核的[长、宽、输入层数、输出层数]，输出的w_shape是以输入为shape的卷积核。
    b = bias_variable(b_shape)  # 不确定要不要加呢，，
    deconved = tf.nn.conv2d_transpose(x,w,output_shape,strides=strides, padding=padding, data_format='NHWC') + b
    return deconved
def dropout(keep_prob, input_feats, train_flag):
    train_flag_tf = tf.cast(train_flag, tf.bool)  # train_flag本来是Python bool，但是tf.cond则需要tf.bool类型。
    if keep_prob < 1:
        output = tf.cond(train_flag_tf, lambda: tf.nn.dropout(input_feats, keep_prob), lambda: input_feats)
    else:
        output = input_feats
    return output
# ---------------------------------------以上是各种基础函数结束------------------------------------- #

# ---------------------------------以下是Resnet网络（带头朝下层的）--------------------------------- #
def conv_block(input_tensor, kernel_size, filters, stage, block, train_flag, stride2=False):
    """conv_block is the block that has a conv layer at shortcut
    conv_block是在shortcut中（后面有两句shortcut的）具有卷积层的结构块（有点像141文中的building block的一层）。
    看完后发现似乎就是对应141论文中表1的101-layer那列的、conv2_x那行的后面那个[]*3中的、[]里面的部分，但是并没有实现那个*3啊。
        以resnet_graph函数中，Stage 2的第一句“x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))”这一句为例。。
    # Arguments
        input_tensor: input tensor
        上句，输入的张量：就是被卷积的张量。本例中输入的张量shape=(?, 128, 128, 64)。
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        上句，输入的核的大小：应该是卷积核的大小，默认为3（确实一般就用3的啊）。本例中输入的就是3。
        filters: list of integers, the nb_filters of 3 conv layer at main path
        上句，输入的滤波器大小：好像是卷积块中各层的通道数。本例就是[64, 64, 256]，所以后面那个nb_filter1/2/3分别是64/64/256。
        stage: integer, current stage label, used for generating layer names
        阶段：输入的当前阶段标签，用来生成层的名称。本例输入的是stage=2。
        block: 'a','b'..., current block label, used for generating layer names
        块：输入的当前块名称，用来生成层的名称。本例输入的是block='a'。
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters  # nb_filter1/2/3分别是64/64/256，三个整数。
    conv_name_base = 'res' + str(stage) + block + '_branch'  # 'res2a_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'  # 'bn2a_branch'

    _, _, _, input_layers = input_tensor.get_shape().as_list()
    if stride2==False:
        stride_for_first_conv = [1, 1, 1, 1]  # 注意，除了这一句用到这个stride_for_first_conv之外，后面shortcut那句也用了。
    else:
        stride_for_first_conv = [1, 2, 2, 1]
    x = conv_layer(input_tensor, [1, 1, input_layers, nb_filter1], [nb_filter1], strides=stride_for_first_conv, padding='SAME', name=conv_name_base + '2a')
    # x = batch_normalization(x, scope=bn_name_base + '2a', is_training=train_flag)
    x = group_norm(x, scope=bn_name_base + '2a')
    # 此时x的shape=(?, 128, 128, 64)，和input_tensor一样。
    """
    以上三句是1*1的卷积（通道数就是输入的nb_filter1），然后批量正则化，然后ReLU激活。
    这对应141论文中表1的101-layer那列的、conv2_x那行的后面那个[]*3的部分中的“1*1,64”的那个层。
    """

    x = conv_layer(x, [kernel_size, kernel_size, nb_filter1, nb_filter2], [nb_filter2], strides=[1, 1, 1, 1], padding='SAME', name=conv_name_base + '2b')
    # x = batch_normalization(x, scope=bn_name_base + '2b', is_training=train_flag)
    x = group_norm(x, scope=bn_name_base + '2b')
    # 此时x的shape仍然=(?, 128, 128, 64)，因为没有池化，且stride一直是1。
    """
    以上三句是3*3的卷积（通道数就是输入的nb_filter2），然后批量正则化，然后ReLU激活。
    这对应141论文中表1的101-layer那列的、conv2_x那行的后面那个[]*3的部分中的“3*3,64”的那个层。
    """

    x = conv_layer(x, [1, 1, nb_filter2, nb_filter3], [nb_filter3], strides=[1, 1, 1, 1], padding='SAME', name=conv_name_base + '2c')
    # x = batch_normalization(x, scope=bn_name_base + '2c', is_training=train_flag, need_relu=False)  # 此时x的shape=(?, 128, 128, 256)  这儿人家本来没激活。
    x = group_norm(x, scope=bn_name_base + '2c', need_relu=False)  # 此时x的shape=(?, 128, 128, 256)  这儿人家本来没激活。
    """
    以上三句是1*1的卷积（通道数就是输入的nb_filter3），然后批量正则化，没有ReLU激活。
    这对应141论文中表1的101-layer那列的、conv2_x那行的后面那个[]*3的部分中的“1*1,256”的那个层。
    """

    shortcut = conv_layer(input_tensor, [1, 1, input_layers, nb_filter3], [nb_filter3], strides=stride_for_first_conv, padding='SAME', name=conv_name_base + '1')
    # 此时shortcut的shape=(?, 128, 128, 256)
    # shortcut = batch_normalization(shortcut, scope=bn_name_base + '1', is_training=train_flag)  # 此时shortcut的shape=(?, 128, 128, 256)
    shortcut = group_norm(shortcut, scope=bn_name_base + '1')  # 此时shortcut的shape=(?, 128, 128, 256)

    x = tf.add(x, shortcut)  # 此时x的shape仍然=(?, 128, 128, 256)，这句是把经过三次卷积后的x和原图卷积过的shortcut相加。
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')  # 此时x的shape仍然=(?, 128, 128, 256)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, train_flag, use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    identity_block是shortcut（对比后面那个conv_block，差别就是没有shortcut）里没有卷积层的块，应该也是141文中的building block的一层。
        然后看到，resnet_graph函数中是先用了一次后面那个conv_block，再用了两次identity_block，
        这说明conv_block和identity_block应该都是对应141论文中表1的101-layer那列的、conv2_x那行的后面那个[]*3中的、[]里面的部分，
        然后总共这三次调用实现了那个*3。
    # Arguments
        下面5个参数，和conv_block中的完全一样。
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 以上三句完全和后面那个conv_block一样

    _, _, _, input_layers = input_tensor.get_shape().as_list()
    x = conv_layer(input_tensor, [1, 1, input_layers, nb_filter1], [nb_filter1], strides=[1, 1, 1, 1], padding='SAME', name=conv_name_base + '2a')
    # x = batch_normalization(x, scope=bn_name_base + '2a', is_training=train_flag)
    x = group_norm(x, scope=bn_name_base + '2a')
    # 以上三句除了第一句没有strides，其他和后面那个conv_block一样。不过conv_block执行的时候strides给设成了1，所以也没区别。

    x = conv_layer(x, [kernel_size, kernel_size, nb_filter1, nb_filter2], [nb_filter2], strides=[1, 1, 1, 1], padding='SAME', name=conv_name_base + '2b')
    # x = batch_normalization(x, scope=bn_name_base + '2b', is_training=train_flag)
    x = group_norm(x, scope=bn_name_base + '2b')
    # 以上三句完全和后面那个conv_block一样

    x = conv_layer(x, [1, 1, nb_filter2, nb_filter3], [nb_filter3], strides=[1, 1, 1, 1], padding='SAME', name=conv_name_base + '2c')
    # x = batch_normalization(x, scope=bn_name_base + '2c', is_training=train_flag, need_relu=False)
    x = group_norm(x, scope=bn_name_base + '2c', need_relu=False)
    # 以上三句完全和后面那个conv_block一样

    x = tf.add(x, input_tensor)
    # 上面这儿是x和input_tensor（原输入张量）相加，而conv_block是把x和输入张量经过卷积和BN后再相加。
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
    return x

def resnet_graph(input_image, architecture, train_flag, stage5=False):  # 残差网络函数，input的shape是(?, 512, 512, 3)
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    """下面Stage 1阶段1，对应141论文表1中conv1那行和conv2_x那行的第一个maxpool部分。"""
    paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])  # 上下左右各补3个0。第一维和第四维不补，所以要有两个[0,0]。
    x = tf.pad(input_image, paddings, "CONSTANT")# 给输入图片的边界补零，此时x的shape是(?, 518, 518, 3)，已验证和KL的一样。<tf.Tensor 'Pad:0' shape=(2, 518, 518, 3) dtype=float32>
    w = weight_variable([7, 7, 3, 64]) # <tf.Variable 'mask_rcnn_model/weights:0' shape=(7, 7, 3, 64) dtype=float32_ref>
    b = bias_variable([64])  #  <tf.Variable 'mask_rcnn_model/biases:0' shape=(64,) dtype=float32_ref>
    x = tf.nn.conv2d(x, w, strides = [1, 2, 2, 1], padding = 'VALID', name = 'conv_1') + b  # <tf.Tensor 'add:0' shape=(2, 256, 256, 64) dtype=float32>

    # 上句，64个卷积核，卷积核的长宽为7，卷积步长为2。现在x的shape应该是(?, 256, 256, 64)，已验证和KL的一样。
    # 【基础】注意那个[64]不能写成64！否则报错说int object is not iterable。这是因为，我用的bias_variable函数里用的是：
    #     initial = tf.constant(0.1, shape=shape)；如果用initializer=tf.constant_initializer(0.1)，似乎就可以用64。
    x = group_norm(x, scope='bn_conv1')  # 正则化，然后ReLU。
    C1 = x = maxpooling(x, [1, 3, 3, 1], [1, 2, 2, 1], name='stage1')  # 此时x的shape=(?, 128, 128, 64)，C1的shape也是这个。已验证和KL的一样。
    # Stage 2
    """
    下面Stage 2阶段2，对应141论文表1中101-layer那列的、conv2_x那行的后面那个[]*3的部分。
    其中用了1次conv_block，2次identity_block，正好是三次、每次先1*1@64再3*3@64最后1*1@256的卷积，就构成了这个conv2_x结构块。
    """
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', train_flag=train_flag)  # 结构块。执行完此句x的shape=(?, 128, 128, 256)，详见conv_block里的注释，已验证和KL的一样。
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_flag=train_flag)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_flag=train_flag)  # 此时x的shape仍然=(?, 128, 128, 256)，C2的shape也是这个，已验证和KL的一样。
    # Stage 3
    """
    下面Stage 3阶段3，对应141论文表1中101-layer那列的、conv3_x那行的后面那个[]*4的部分。
    其中用了1次conv_block，3次identity_block，正好是4次、每次先1*1@128再3*3@128最后1*1@512的卷积，就构成了这个conv3_x结构块。
    """
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_flag=train_flag, stride2=True)  # 执行此句后x的shape=(?, 64, 64, 512)，因为这里strides用的是原来的2。
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_flag=train_flag)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_flag=train_flag)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_flag=train_flag)  # 此时x的shape仍然=(?, 64, 64, 512)，C3的shape也是这个，已验证和KL的一样。
    # Stage 4
    """
    下面Stage 4阶段4，对应141论文表1中101-layer那列的、conv4_x那行的后面那个[]*23的部分。
    其中用了1次conv_block，22次identity_block，正好是23次、每次先1*1@256再3*3@256最后1*1@1024的卷积，就构成了这个conv4_x结构块。
    """
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_flag=train_flag, stride2=True)  # 执行此句后x的shape=(?, 32, 32, 1024)，因为这里strides用的是原来的2。
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]  # block_count=22，即，下句for循环的range是0~22，正好是加23个层。
    for i in range(block_count):  # 加22个层，都是用identity_block函数加。就是141论文中表1的101-layer那列的、conv4_x那行
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_flag=train_flag)
    C4 = x  # 此时x和C4的hape=(?, 32, 32, 1024)，已验证和KL的一样。
    # Stage 5
    """
    下面Stage 5阶段5，对应141论文表1中101-layer那列的、conv5_x那行的后面那个[]*3的部分。
    其中用了1次conv_block，2次identity_block，正好是3次、每次先1*1@2512再3*3@512最后1*12048的卷积，就构成了这个conv5_x结构块。
    """
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_flag=train_flag, stride2=True)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_flag=train_flag)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_flag=train_flag)  # C5的shape=(?, 16, 16, 2048)，已验证和KL的一样。
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

# 下面是原来程序里的top_down_layers。。结果神奇的是，发现换回这个，那些loss居然至少神奇地不是nan了，而且前面三项损失也没那么大了（后面两项损失仍然是0，似乎有问题）。。。这是个什么节奏啊。。。
import keras.layers as KL
def top_down_layers(C5, C4, C3, C2):
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)  # P5的shape是shape=(?, 16, 16, 256)，因为卷积是1*1的，只改了通道数。
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])  # P4的shape是shape=(?, 32, 32, 256)
    """上句，
    KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5)是把P5上采样，得到shape=(?, 32, 32, 256)的张量。
    KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)是把C4做1*1卷积，也得到shape=(?, 32, 32, 256)的张量。
    然后二者相加。
    所以这个P4相当于就是加入了不同尺度的特征了，反卷积升维再相加（也有用拼接的），挺常用的。。
    """
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])  # P3的shape=(?, 64, 64, 256)
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])  # P2的shape=(?, 128, 128, 256)
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 以上4句的卷积，都没有改变P5~P2的shape。
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)  # P6的shape=(?, 8, 8, 256)
    """以上，应该是完成了各种特征提取了。想想也确实，就执行了一次特征提取。。"""
    return [P6, P5, P4, P3, P2]
# ---------------------------------以上Resnet网络（带头朝下层的）结束--------------------------------- #

# ----------------------------------下面是用来对比的一些特征提取网络----------------------------------- #
# 1、VGG19相关。
def VGG19(input_image, train_flag):
    """类比于上面的Resnet的输入，就是原图和是否训练的标签。然后VGG19本来是16个卷积层+3个全连接层，但是那三个全连接层输出的都是展平了的，
        似乎是没有什么作用，所以这儿就把它们删掉了，只保留前面卷积层了。
    输出则是类似于C2~C5的那几个大小的东西，可以把通道数变一下吧。
    需要考虑一个问题，就是如果直接从头训练的话，可能会出nan，如果这样的话，解决方法如下：
    1，可以考虑初始化为he_normalizer，
    2，可以考虑恢复原来的东西（这个有点困难，因为，卷积层的大小可能和原来的不一样的）。。。"""
    b, h, w, c = input_image.get_shape().as_list()
    x1_1 = conv_layer(input_image, [1, 1, c, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
    x1_1_o = tf.nn.relu(batch_normalization(x1_1, scope='bn1_1', is_training=train_flag))
    x1_2 = conv_layer(x1_1_o, [1, 1, 64, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
    x1_2_o = tf.nn.relu(batch_normalization(x1_2, scope='bn1_2', is_training=train_flag))
    C1 = maxpooling(x1_2_o, [1, 2, 2, 1], [1, 4, 4, 1], "pool1")  # (?, 128, 128, 64)
    x2_1 = conv_layer(C1, [3, 3, 64, 128], [128], strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
    x2_1_o = tf.nn.relu(batch_normalization(x2_1, scope='bn2_1', is_training=train_flag))
    x2_2 = conv_layer(x2_1_o, [3, 3, 128, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
    x2_2_o = tf.nn.relu(batch_normalization(x2_2, scope='bn2_2', is_training=train_flag))
    C2 = x2_2_o # 不用maxpooling(x2_2_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool2")了，因为想要的C2的shape是(?, 128, 128, 256)。
    x3_1 = conv_layer(C2, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
    x3_1_o = tf.nn.relu(batch_normalization(x3_1, scope='bn3_1', is_training=train_flag))
    x3_2 = conv_layer(x3_1_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
    x3_2_o = tf.nn.relu(batch_normalization(x3_2, scope='bn3_2', is_training=train_flag))
    x3_3 = conv_layer(x3_2_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
    x3_3_o = tf.nn.relu(batch_normalization(x3_3, scope='bn3_3', is_training=train_flag))
    x3_4 = conv_layer(x3_3_o, [3, 3, 256, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv3_4')
    x3_4_o = tf.nn.relu(batch_normalization(x3_4, scope='bn3_4', is_training=train_flag))
    C3 = maxpooling(x3_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool3")  # (?, 64, 64, 512)
    x4_1 = conv_layer(C3, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_1')
    x4_1_o = tf.nn.relu(batch_normalization(x4_1, scope='bn4_1', is_training=train_flag))
    x4_2 = conv_layer(x4_1_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_2')
    x4_2_o = tf.nn.relu(batch_normalization(x4_2, scope='bn4_2', is_training=train_flag))
    x4_3 = conv_layer(x4_2_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_3')
    x4_3_o = tf.nn.relu(batch_normalization(x4_3, scope='bn4_3', is_training=train_flag))
    x4_4 = conv_layer(x4_3_o, [3, 3, 512, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv4_4')
    x4_4_o = tf.nn.relu(batch_normalization(x4_4, scope='bn4_4', is_training=train_flag))
    C4 = maxpooling(x4_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool4")  # (?, 32, 32, 1024)
    x5_1 = conv_layer(C4, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_1')
    x5_1_o = tf.nn.relu(batch_normalization(x5_1, scope='bn5_1', is_training=train_flag))
    x5_2 = conv_layer(x5_1_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_2')
    x5_2_o = tf.nn.relu(batch_normalization(x5_2, scope='bn5_2', is_training=train_flag))
    x5_3 = conv_layer(x5_2_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_3')
    x5_3_o = tf.nn.relu(batch_normalization(x5_3, scope='bn5_3', is_training=train_flag))
    x5_4 = conv_layer(x5_3_o, [3, 3, 1024, 2048], [2048], strides=[1, 1, 1, 1], padding='SAME', name='conv5_4')
    x5_4_o = tf.nn.relu(batch_normalization(x5_4, scope='bn5_4', is_training=train_flag))
    C5 = maxpooling(x5_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool5")  # (?, 16, 16, 2048)
    return [C1, C2, C3, C4, C5]

# 2、VGG19-FCN相关。
def VGG19_FCN(input_image, train_flag):
    """原来VGG19最后三层全连接层，输出的都是(批大小, 4096或者类别数)的、展平了的特征，没用，所以上面函数就直接丢弃了这三个全连接层。
    全卷积的话，就让他输出(批大小, 16, 16, 2048)的东西，和C5一样的shape，把它当做C5试试看吧。"""
    b, h, w, c = input_image.get_shape().as_list()
    x1_1 = conv_layer(input_image, [1, 1, c, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
    x1_1_o = tf.nn.relu(batch_normalization(x1_1, scope='bn1_1', is_training=train_flag))
    x1_2 = conv_layer(x1_1_o, [1, 1, 64, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
    x1_2_o = tf.nn.relu(batch_normalization(x1_2, scope='bn1_2', is_training=train_flag))
    C1 = maxpooling(x1_2_o, [1, 2, 2, 1], [1, 4, 4, 1], "pool1")  # (?, 128, 128, 64)
    x2_1 = conv_layer(C1, [3, 3, 64, 128], [128], strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
    x2_1_o = tf.nn.relu(batch_normalization(x2_1, scope='bn2_1', is_training=train_flag))
    x2_2 = conv_layer(x2_1_o, [3, 3, 128, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
    x2_2_o = tf.nn.relu(batch_normalization(x2_2, scope='bn2_2', is_training=train_flag))
    C2 = x2_2_o  # 不用maxpooling(x2_2_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool2")了，因为想要的C2的shape是(?, 128, 128, 256)。
    x3_1 = conv_layer(C2, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
    x3_1_o = tf.nn.relu(batch_normalization(x3_1, scope='bn3_1', is_training=train_flag))
    x3_2 = conv_layer(x3_1_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
    x3_2_o = tf.nn.relu(batch_normalization(x3_2, scope='bn3_2', is_training=train_flag))
    x3_3 = conv_layer(x3_2_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
    x3_3_o = tf.nn.relu(batch_normalization(x3_3, scope='bn3_3', is_training=train_flag))
    x3_4 = conv_layer(x3_3_o, [3, 3, 256, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv3_4')
    x3_4_o = tf.nn.relu(batch_normalization(x3_4, scope='bn3_4', is_training=train_flag))
    C3 = maxpooling(x3_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool3")  # (?, 64, 64, 512)
    x4_1 = conv_layer(C3, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_1')
    x4_1_o = tf.nn.relu(batch_normalization(x4_1, scope='bn4_1', is_training=train_flag))
    x4_2 = conv_layer(x4_1_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_2')
    x4_2_o = tf.nn.relu(batch_normalization(x4_2, scope='bn4_2', is_training=train_flag))
    x4_3 = conv_layer(x4_2_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_3')
    x4_3_o = tf.nn.relu(batch_normalization(x4_3, scope='bn4_3', is_training=train_flag))
    x4_4 = conv_layer(x4_3_o, [3, 3, 512, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv4_4')
    x4_4_o = tf.nn.relu(batch_normalization(x4_4, scope='bn4_4', is_training=train_flag))
    C4 = maxpooling(x4_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool4")  # (?, 32, 32, 1024)
    x5_1 = conv_layer(C4, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_1')
    x5_1_o = tf.nn.relu(batch_normalization(x5_1, scope='bn5_1', is_training=train_flag))
    x5_2 = conv_layer(x5_1_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_2')
    x5_2_o = tf.nn.relu(batch_normalization(x5_2, scope='bn5_2', is_training=train_flag))
    x5_3 = conv_layer(x5_2_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_3')
    x5_3_o = tf.nn.relu(batch_normalization(x5_3, scope='bn5_3', is_training=train_flag))
    x5_4 = conv_layer(x5_3_o, [3, 3, 1024, 2048], [2048], strides=[1, 1, 1, 1], padding='SAME', name='conv5_4')
    x5_4_o = tf.nn.relu(batch_normalization(x5_4, scope='bn5_4', is_training=train_flag))
    C5_pre = maxpooling(x5_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool5")  # (?, 16, 16, 2048)
    b, h, w, c = C5_pre.get_shape().as_list()
    """以下是三个代替全连接的卷积。采用same的padding。见《E:/赵屾的文件/61-FCN语义分割/FCN》里的FCN.py。"""
    x6 = conv_layer(C5_pre, [h, w, c, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv6')
    x6_o = tf.nn.relu(batch_normalization(x6, scope='bn6', is_training=train_flag))  # 应该还是(?, 16, 16, 512)
    x7 = conv_layer(x6_o, [1, 1, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv7')
    x7_o = tf.nn.relu(batch_normalization(x7, scope='bn7', is_training=train_flag))  # 应该还是(?, 16, 16, 512)
    x8 = conv_layer(x7_o, [1, 1, 512, c], [c], strides=[1, 1, 1, 1], padding='SAME', name='conv8')
    C5 = tf.nn.relu(batch_normalization(x8, scope='bn8', is_training=train_flag))  # 应该还是(?, 16, 16, 2048)
    """有点讨厌的是，加了这三个卷积层，就莫名其妙OOM了。只好用服务器跑。不过这就说明，看来VGG19_FCN的参数不比Resnet少啊。"""
    return [C1, C2, C3, C4, C5]

def VGG19_FCN_simp(input_image, train_flag):
    """就是那个和[h, w, c, 512]的卷积核去卷积的时候，经常OOM，所以简化一下，改成3个3*3的好了。其实感受野还是不对的，
    原来是一个15*15的，感受野是15*15；现在是3个3*3，感受野是7*7，如果想要15*15的话就应该7个3*3的才行。不过不想管他了。"""
    b, h, w, c = input_image.get_shape().as_list()
    x1_1 = conv_layer(input_image, [1, 1, c, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
    x1_1_o = tf.nn.relu(batch_normalization(x1_1, scope='bn1_1', is_training=train_flag))
    x1_2 = conv_layer(x1_1_o, [1, 1, 64, 64], [64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
    x1_2_o = tf.nn.relu(batch_normalization(x1_2, scope='bn1_2', is_training=train_flag))
    C1 = maxpooling(x1_2_o, [1, 2, 2, 1], [1, 4, 4, 1], "pool1")  # (?, 128, 128, 64)
    x2_1 = conv_layer(C1, [3, 3, 64, 128], [128], strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
    x2_1_o = tf.nn.relu(batch_normalization(x2_1, scope='bn2_1', is_training=train_flag))
    x2_2 = conv_layer(x2_1_o, [3, 3, 128, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
    x2_2_o = tf.nn.relu(batch_normalization(x2_2, scope='bn2_2', is_training=train_flag))
    C2 = x2_2_o  # 不用maxpooling(x2_2_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool2")了，因为想要的C2的shape是(?, 128, 128, 256)。
    x3_1 = conv_layer(C2, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
    x3_1_o = tf.nn.relu(batch_normalization(x3_1, scope='bn3_1', is_training=train_flag))
    x3_2 = conv_layer(x3_1_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
    x3_2_o = tf.nn.relu(batch_normalization(x3_2, scope='bn3_2', is_training=train_flag))
    x3_3 = conv_layer(x3_2_o, [3, 3, 256, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
    x3_3_o = tf.nn.relu(batch_normalization(x3_3, scope='bn3_3', is_training=train_flag))
    x3_4 = conv_layer(x3_3_o, [3, 3, 256, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv3_4')
    x3_4_o = tf.nn.relu(batch_normalization(x3_4, scope='bn3_4', is_training=train_flag))
    C3 = maxpooling(x3_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool3")  # (?, 64, 64, 512)
    x4_1 = conv_layer(C3, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_1')
    x4_1_o = tf.nn.relu(batch_normalization(x4_1, scope='bn4_1', is_training=train_flag))
    x4_2 = conv_layer(x4_1_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_2')
    x4_2_o = tf.nn.relu(batch_normalization(x4_2, scope='bn4_2', is_training=train_flag))
    x4_3 = conv_layer(x4_2_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_3')
    x4_3_o = tf.nn.relu(batch_normalization(x4_3, scope='bn4_3', is_training=train_flag))
    x4_4 = conv_layer(x4_3_o, [3, 3, 512, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv4_4')
    x4_4_o = tf.nn.relu(batch_normalization(x4_4, scope='bn4_4', is_training=train_flag))
    C4 = maxpooling(x4_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool4")  # (?, 32, 32, 1024)
    x5_1 = conv_layer(C4, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_1')
    x5_1_o = tf.nn.relu(batch_normalization(x5_1, scope='bn5_1', is_training=train_flag))
    x5_2 = conv_layer(x5_1_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_2')
    x5_2_o = tf.nn.relu(batch_normalization(x5_2, scope='bn5_2', is_training=train_flag))
    x5_3 = conv_layer(x5_2_o, [3, 3, 1024, 1024], [1024], strides=[1, 1, 1, 1], padding='SAME', name='conv5_3')
    x5_3_o = tf.nn.relu(batch_normalization(x5_3, scope='bn5_3', is_training=train_flag))
    x5_4 = conv_layer(x5_3_o, [3, 3, 1024, 2048], [2048], strides=[1, 1, 1, 1], padding='SAME', name='conv5_4')
    x5_4_o = tf.nn.relu(batch_normalization(x5_4, scope='bn5_4', is_training=train_flag))
    C5_pre = maxpooling(x5_4_o, [1, 2, 2, 1], [1, 2, 2, 1], "pool5")  # (?, 16, 16, 2048)
    b, h, w, c = C5_pre.get_shape().as_list()
    """以下是三个代替全连接的卷积。采用same的padding。见《E:/赵屾的文件/61-FCN语义分割/FCN》里的FCN.py。"""
    x6 = conv_layer(C5_pre, [3, 3, c, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv6')  # 用h*w的容易OOM，改成3*3的代替好了，虽然减小了感受野
    x6_o = tf.nn.relu(batch_normalization(x6, scope='bn6', is_training=train_flag))  # 应该还是(?, 16, 16, 512)
    x7 = conv_layer(x6_o, [3, 3, 512, 512], [512], strides=[1, 1, 1, 1], padding='SAME', name='conv7')
    x7_o = tf.nn.relu(batch_normalization(x7, scope='bn7', is_training=train_flag))  # 应该还是(?, 16, 16, 512)
    x8 = conv_layer(x7_o, [3, 3, 512, c], [c], strides=[1, 1, 1, 1], padding='SAME', name='conv8')
    C5 = tf.nn.relu(batch_normalization(x8, scope='bn8', is_training=train_flag))  # 应该还是(?, 16, 16, 2048)
    """有点讨厌的是，加了这三个卷积层，就莫名其妙OOM了。只好用服务器跑。不过这就说明，看来VGG19_FCN的参数不比Resnet少啊。"""
    return [C1, C2, C3, C4, C5]

# 3、GoogLeNet相关。不过此网络似乎已经淘汰了。。
import GoogLeNet_for_FEN
def GoogleNet_FCN(config, input_image):
    """那个GoogleNet的网络，试着用来提取特征。"""
    _, h, w, c = input_image.get_shape().as_list()
    weights = {
        'conv1_7x7_S2': tf.get_variable('w1', shape=[7, 7, c, 64], initializer=tf.contrib.keras.initializers.he_normal()),  # 这个是输入图像的第一次卷积，
        'conv2_1x1_S1': tf.get_variable('w2', shape=[1, 1, 64, 64], initializer=tf.contrib.keras.initializers.he_normal()),
        'conv2_3x3_S1': tf.get_variable('w3', shape=[3, 3, 64, 192], initializer=tf.contrib.keras.initializers.he_normal()),
        'FC2': tf.Variable(tf.get_variable('w4', shape=[16 * 16 * 1024, config.NUM_CLASSES], initializer=tf.contrib.keras.initializers.he_normal()))
        # 上面那个16*16*1024，还得注意需要和GoogleLeNet_topological_structure里的softmax2的shape一致，挺弱智的玩意。。。不懂原作者为什么这么写。。
    }
    biases = {
        'conv1_7x7_S2': tf.Variable(tf.constant(0.1, shape=[64])),
        'conv2_1x1_S1': tf.Variable(tf.constant(0.1, shape=[64])),
        'conv2_3x3_S1': tf.Variable(tf.constant(0.1, shape=[192])),
        'FC2': tf.Variable(tf.constant(0.1, shape=[config.NUM_CLASSES]))
    }
    [C1, C2, C3, C4, C5], _ = GoogLeNet_for_FEN.GoogleLeNet_topological_structure(input_image, weights, biases)
    return [C1, C2, C3, C4, C5]

# 4、densenet相关。略复杂一些。
def densenet(input_image, train_flag):
    """densenet网络"""
    # densenet_para = {'growth_rate': [32, 32, 32, 32, 32],  # 原来是12，效果不好，试试32怎么样。
    #                  'depth': 121,
    #                  'total_blocks': 4,
    #                  'layers_per_block': [6, 12, 24, 16],
    #                  'keep_prob': 1,  # dropout的保持率，如果不想用dropout就把keep_prob设成1。他原来是0.5
    #                  'bc': True,  # 是否加瓶颈层。
    #                  'reduction': 0.5,  # 他原来是0.5，就是每个transition_layer把通道数也减小一半。
    #                  # 发现0.5和growth_rate=32、layers_per_block = [6, 12, 24, 16]配合，前面C2~C4和Resnet的形状是一样的。
    #                  'embedding_dim': 64
    #                  'use_feats_after_trans': False
    #                  }
    """上面这个注释掉的，是Densenet论文中给出来的参数（Densenet121）。但是结果不太好，甚至还不如VGG19。
    后面程序，所有注释中的shape都是按照上面的配置弄的，后面换配置的时候就没再改了。
    又加了一个use_feats_after_trans，如果要这个的话，那么就是在transition层后的东西作为输出特征，否则是transition前的作为特征。
    后面程序，所有注释中的shape都是按照'use_feats_after_trans': False弄的，不过后来觉得似乎还是True更靠谱一点，否则，有那么几个通道
        就是直接把输入拼过去的，连卷积都没有，看上去不怎么靠谱。
    """
    densenet_para = {'growth_rate': [64, 96, 128, 256, 512],
                     'depth': 121,  # 这个设置里，depth其实是没用的。
                     'total_blocks': 4,
                     'layers_per_block': [2, 2, 2, 2],
                     'keep_prob': 1,  # dropout的保持率，如果不想用dropout就把keep_prob设成1。他原来是0.5
                     'bc': True,  # 是否加瓶颈层。
                     'reduction': 1,  # 现在不减小了。
                     'embedding_dim': 64,
                     'use_feats_after_trans': True
                     }
    growth_rate = densenet_para['growth_rate']
    total_blocks = densenet_para['total_blocks']
    # layers_per_block = (densenet_para['depth'] - (total_blocks + 1)) // total_blocks
    layers_per_block = densenet_para['layers_per_block']  # 和论文142的DenseNet121一样。
    # layers_per_block = [6, 12, 24, 16]  # 和论文142的DenseNet121一样。
    use_feats_after_trans = densenet_para['use_feats_after_trans']
    assert total_blocks==len(layers_per_block), 'layers_per_block的长度，要和总的卷积块数一致。'
    assert total_blocks==len(growth_rate) - 1, 'growth_rate的长度，要比总的卷积块数多1。'
    b, h, w, c = input_image.get_shape().as_list()
    with tf.variable_scope("Initial_convolution"):
        w = weight_variable([5, 5, c, growth_rate[0] * 2])  # densenet是growth_rate * 2
        b = bias_variable([growth_rate[0] * 2])  # growth_rate * 2
        x = tf.nn.conv2d(input_image, w, strides=[1, 1, 1, 1], padding='SAME', name='conv_1') + b  # shape=(2, 512, 512, 64)
        x = maxpooling(x, [1, 2, 2, 1], [1, 2, 2, 1], "pool1")  # shape=(2, 256, 256, 64)
    with tf.variable_scope("C1"):
        w1 = weight_variable([3, 3, growth_rate[0] * 2, 64])
        b1 = bias_variable([64])
        x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1') + b1  # shape=(2, 256, 256, 64)
    if not use_feats_after_trans:
        C1 = x = maxpooling(x, [1, 2, 2, 1], [1, 2, 2, 1], "poolC1")  # shape=(2, 128, 128, 64)
    else:
        C1 = x
    out_feats = []
    for block in range(total_blocks):
        # block应该就是循环变量，total_blocks是总块数，主函数里定义的是3个。
        with tf.variable_scope("Block_%d" % block):
            x = add_block(densenet_para, x, growth_rate[block+1], layers_per_block[block], train_flag)
            # 上句，加入一个密集块，就是增加了通道数。并没有池化改变分辨率。第一次执行for循环，这儿的shape是(2, 128, 128, 256)；
            # 第二次执行for，这儿的shape是(2, 64, 64, 512)；第三次执行for，这儿的shape=(2, 32, 32, 1024)；
            # 第四次执行for，这儿的shape=(2, 16, 16, 1024)。
        if not use_feats_after_trans:
            out_feats.append(x)
        if block != total_blocks - 1:
            with tf.variable_scope("Transition_after_block_%d" % block):
                x = transition_layer(densenet_para, x, train_flag)
                # 此函数的功能是降维和池化，长宽减掉一半，通道数变成reduction倍（如果reduction=0.5则是减掉一半）。
            if use_feats_after_trans:
                out_feats.append(x)
    if not use_feats_after_trans:
        [C2, C3, C4, C5] = out_feats
    else:
        [C2, C3, C4] = out_feats
        C5 = transition_layer(densenet_para, x, train_flag)
    with tf.variable_scope("Feature_embeddings"):
        embeddings = final_output(densenet_para, x, train_flag)  # 【第三步】这似乎是传说中的“嵌入”。
        # 好像是前面那一堆（卷积啊、升降维啊、池化啊）做完了之后得到那个output（它的shape=(?, 8, 8, 132)），再在最后用那个final_output啊。
        # 【重要总结！！】
        # 看完了【第一步】、【第二步】和【第三步】，似乎感觉上面的过程就是构造那个网络的，卷积-池化-激活-全连接什么的。下面loss那句才是按照论文里的内容去计算那个损失函数的。
        # 只不过这个做法和常用的似乎不太一样啊，是初次卷积→加块（加了3次，每次加6个块，每个块有12个层，层就是通道数
        # 每个层都是先归一化，然后激活、卷积、Dropout（这儿不池化，等到这次加块完了，再池化）。
        # →降维池化（前两次加块的时候）-最终嵌入（图像归一化、激活、池化、展平、全连接矩阵相乘、l2正则化）。
        # 那个l2正则化，和论文中说的一样，论文3.5节一开始说在计算损失函数之前，用l2正则化对嵌入向量做了正则化。
    return [C1, C2, C3, C4, C5], embeddings

def add_block(densenet_para, input_feats, growth_rate, layers_per_block, train_flag):
    x = input_feats  # 先把输出设定为输入，再加一些层
    for layer in range(layers_per_block):
    # layer是计数变量，layers_per_block是每个卷积块里的卷积层数。然后每个块里的层数，是随着块的个数变的，但是总的层数似乎是40，不变的。
        with tf.variable_scope("layer_%d" % layer):
            x = add_internal_layer(densenet_para, x, growth_rate, train_flag)  # 加上growth_rate个通道
    return x

def add_internal_layer(densenet_para, input_feats, growth_rate, train_flag):
    if not densenet_para['bc']:  # 主函数里是'bc_mode': True,。
        comp_out = composite_function(densenet_para, input_feats, out_features=growth_rate, kernel_size=3, train_flag=train_flag)
        # 上句，组成函数，先BN，再RELU（激活），再卷积，最后dropout。
    else:  # 如果'bc'是True，就先升维再降维。
        # bottleneck_out = bottleneck(densenet_para, input_feats, out_features=growth_rate)
        # 上句，就是先用1*1的卷积折腾一下，升维然后再降维回来。后来想，索性改成3*3的得了，然后中间层的通道数不要那么多。所以有了下句：
        bottleneck_out = bottleneck(densenet_para, input_feats, kernel_size=3, out_features=growth_rate)
        comp_out = composite_function(densenet_para, bottleneck_out, out_features=growth_rate, kernel_size=3, train_flag=train_flag)
        # 上句，组成函数，同上面。
    # 总之，这个if-else就是构造一个growth_rate（32）通道的张量，无论bc_mode是真是假，输出都是(?, h, w, 32)。
    # output = tf.concat(axis=3, values=(input_feats, comp_out))
    # 上句，把组成函数（composite function）的输入输出拼接起来。最终输出output是shape=(?, h, h, 输入通道数+32)的张量。
    # 【总结】执行一次这个add_internal_layer，不改变原来输入的h和w，而只是先把输入的东西经过两个卷积层（一个1*1的一个3*3的），
    #     并且把这两个卷积层最后的输出通道数是32（growth_rate），然后把这个输出拼接到原来输入上。
    # 其实也就是说，这个函数执行一次，原输入所占的通道数，还是比经过卷积层处理后的通道数要大多了。
    # 所以，这么想来，这个东西用处会很大吗？如果用原输入真有这么好，那还要卷积层干个屁啊。。。
    # 他这儿也说，https://www.reddit.com/r/MachineLearning/comments/67fds7/d_how_does_densenet_compare_to_resnet_and/
    # 可以try a "Wide DenseNet", by making it shallow (set the depth to be smaller) and wide (set the growthRate k to be larger).
    # 然而，试了上面的方法，还是不行，妈的。。索性试试把这个concat不要掉了，我他妈的不要他那个输入了好吧，就算是更宽的，growth_rate和
    #     输入的张量通道数一样的，那么还是有一半的通道数是没有经过卷积的，这他妈的不是自废武功吗。。。索性试试放弃那个拼接的，见下：
    _, _, _, c_in = input_feats.get_shape().as_list()
    _, _, _, c_inter = comp_out.get_shape().as_list()
    c_out = c_in + c_inter
    output = conv_layer(comp_out, [3, 3, c_inter, c_out], [c_out], strides=[1, 1, 1, 1], padding='SAME', name='imp')
    return output

def composite_function(densenet_para, input_feats, out_features, kernel_size=3, train_flag=True):
    # 那个out_features是输出图像通道数。
    _, _, _, in_features = input_feats.get_shape().as_list()
    with tf.variable_scope("composite_function"):
        output = batch_normalization(input_feats, scope='cf', is_training=train_flag, need_relu=True)
        output = conv_layer(output, [kernel_size, kernel_size, in_features, out_features], [out_features], strides=[1, 1, 1, 1], padding='SAME', name='c')
        # output = dropout(densenet_para['keep_prob'], output, train_flag)  # 先不要dropout了
        # output = conv_layer(input_feats, [kernel_size, kernel_size, in_features, out_features], [out_features], strides=[1, 1, 1, 1], padding='SAME', name='c')
        # output = batch_normalization(output, scope='cf', is_training=train_flag, need_relu=True)  # 弄成先卷积再bn的试试。。
        # 最终输出output是shape=(?, 32, 32, 12)的张量；而输入是shape=(?, 32, 32, 48)的张量。
        # 说明这个函数就是升降维用的，那个12是因为out_features是growth_rate，即12啊。
    return output

def bottleneck(densenet_para, input_feats, out_features, kernel_size=1, train_flag=True):
    # 所谓bottleneck，是在昂贵的并行卷积（指的是3×3或者5×5卷积）之前使用了1x1卷积块(NIN)来减少特征的数量
    # http://m.blog.csdn.net/qq_31531635/article/details/72822085
    # 但，好像和现在用的不太一样啊？？？？
    # 发现它和前面的composite_function差别就是输出特征一个是12一个是48，然后卷积核一个是3一个是1啊。
    _, _, _, in_features = input_feats.get_shape().as_list()
    with tf.variable_scope("bottleneck"):
        output = batch_normalization(input_feats, scope='bc', is_training=train_flag, need_relu=True)
        # inter_features = out_features * 4  # 卷积后的通道数，因为out_features是growth_rate（即12），所以卷积后的通道数是48。
        inter_features = out_features  # 但我为什么要乘以4呢，反正还要折腾回去，干什么呢，有那些参数，都够弄一个3*3的卷积了，不乘试试。。。
        output = conv_layer(output, [kernel_size, kernel_size, in_features, inter_features], [inter_features], strides=[1, 1, 1, 1], padding='SAME', name='bc')
        # 上句进行卷积，out_features=inter_features是卷积层输出的通道数。然后卷积核大小是1啊。
        # 卷积核大小为1的卷积，看这个网址：https://www.zhihu.com/question/56024942
        # 重点在于，“降维（ dimension reductionality ）。比如，一张500*500且厚度depth为100的图片，
        # 在20个filter上做1*1的卷积，那么结果的大小为500*500*20。就相当于把100个特征降维成了20个了”
        # 以及，“卷积的输入输出是长方体【卷积的输入输出不只是一张图，而是有很多通道的】，所以1*1卷积实际上就是对每个像素点，
        # 在不同的通道上进行线性组合（信息整合），且保留了图片的原有平面结构，而只是调整depth（从而完成升维或降维的功能）。”
        # 在这个程序中的话，好像也是这个用法，因为是设了那个“卷积层输出的通道数”啊，然后1*1的卷积就是多个通道
        # （可以理解维特征通道，每个通道数都是一种特征）线性叠加。
        # 对于1*1的卷积，SAME或者VALID应该是没什么关系的。
        # output = dropout(densenet_para['keep_prob'], output, train_flag)  # 先不要dropout了

        # inter_features = out_features * 4  # 卷积后的通道数，因为out_features是growth_rate（即12），所以卷积后的通道数是48。
        # output = conv_layer(input_feats, [kernel_size, kernel_size, in_features, inter_features], [inter_features], strides=[1, 1, 1, 1], padding='VALID', name='bc')
        # output = batch_normalization(output, scope='bc', is_training=train_flag, need_relu=True)
    return output

def transition_layer(densenet_para, input_feats, train_flag=True):
    out_features = int(int(input_feats.get_shape()[-1]) * densenet_para['reduction'])
    # 上句，好像是把输出通道数设成了输入通道数乘以那个reduction（0.5），相当于是降维了。
    output = composite_function(densenet_para, input_feats, out_features=out_features, kernel_size=1, train_flag=train_flag)
    output = avg_pool(output, k=2)
    return output

def final_output(densenet_para, input_feats, train_flag=True):
    output = batch_normalization(input_feats, scope='fo', is_training=train_flag, need_relu=True)
    last_pool_kernel = int(output.get_shape()[-2])  # last_pool_kernel是8
    output = avg_pool(output, k=last_pool_kernel)
    features_total = int(output.get_shape()[-1])  # 132
    output = tf.reshape(output, [-1, features_total])
    W = weight_variable([features_total, densenet_para['embedding_dim']])
    bias = bias_variable([densenet_para['embedding_dim']])  # embedding_dim是64
    output = tf.matmul(output, W) + bias
    output = tf.nn.l2_normalize(output,1)
    return output