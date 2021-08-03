#coding=utf-8
import tensorflow as tf
import MaskRCNN_2_ResNet_and_other_FEN as MaskRCNN_2_ResNet
import dict_learning_manners
import MRCNN_utils
import LISTA_manners
def pyramidroialign(rois, feature_maps, pool_shape, image_shape, name):
    """
    输入参数：
        rois: MaskRCNN_5_find_gt.DetectionTargetLayer函数弄出来的、一个批次中的所有样例（含有正例和负例），shape是(4, ?, ?)，即
            (批大小, 一张图中提出数config.TRAIN_ROIS_PER_IMAGE,4)。记住，这是4个config.TRAIN_ROIS_PER_IMAGE*4的矩阵，
            每个矩阵就是一张图，每个矩阵都有config.TRAIN_ROIS_PER_IMAGE行，每一行的4个数就是这个样例的归一化了的角点坐标。
            <tf.Tensor 'mask_rcnn_model/rois:0' shape=(2, ?, ?) dtype=float32>，说明是归一化了的rois。
        feature_maps: 不同尺度的特征图；MaskMRCNN_model里的mrcnn_feature_maps, 即那个[P2, P3, P4, P5]
        pool_shape:  池化大小，应该是这个函数输出的特征的长宽
        image_shape:  原图长宽
        name:  暂时没用。
    输出参数：
        pooled: 对每个正例和负例的、在不同尺度提取的特征。shape=(1, ?, 7, 7, 256)，即(1, 这个批次所有图中所有样例的个数, 特征长, 特征宽, 特征通道数)
    """
    pool_shape = tuple(pool_shape)  # (7, 7)
    image_shape = tuple(image_shape)  # (512, 512, 3)

    y1, x1, y2, x2 = tf.split(rois, 4, axis=2)  # 在第2维分开那些boxes，这是说分开四个角点了。
    # 执行后，y1的shape仍然是(4, ?, ?)，第一个4是说一个批次里4张图（batch_slice的结果），第二个?是一张图中提出数，
    #     第三个?是原来第三个?除以4（原来第三个?应该是4，那现在就是1了，这是合理的，因为原来是4个角点，现在是其中1个角点，而且这一维后面就给squeeze掉了）。
    h = y2 - y1  # h的shape=(4, ?, ?)。第一个4和第二个?意义同前，第三个?就是1了（表示某一个正例或者负例的长）
    w = x2 - x1  # 类似，某个正例或者负例的宽。
    # 以上，弄出来图像的长宽。
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    # 上句，计算图像总的面积（不是某个提出的面积）。
    roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))  # roi_level的shape=(4,?,?)。第三个?就是1。
    # 上句执行完了，roi_level是<tf.Tensor 'mask_rcnn_model/truediv_32:0' shape=(2, ?, ?) dtype=float32>
    roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))  # shape=(4,?,?)
    # 上句执行完了，roi_level是<tf.Tensor 'mask_rcnn_model/Minimum_16:0' shape=(2, ?, ?) dtype=int32>
    roi_level = tf.squeeze(roi_level, 2)  # shape=(4,?)
    # 上句执行完了，roi_level是<tf.Tensor 'mask_rcnn_model/Squeeze_2:0' shape=(2, ?) dtype=int32>
    # 以上3句，来判断每个提出该用哪一层的输出，roi_level是几就用P几，大尺度ROI就用shape大一些的金字塔层，比如P2；小尺度ROI就用小一些的特征层，比如P5。
    # roi_level最终的shape=(4,?)，那个4是一个批次里有4张图，?是一张图里的正例和负例数（就是config.TRAIN_ROIS_PER_IMAGE=150了）。
    #     即，roi_level是个4*150的矩阵，每一行代表一张图，一行有150个数，每个数就代表这张图中的某个正例或者负例的尺度值（是个2~6之间的数）。
    """上面这一大段都是用来根据boxes（也就是rois的外接矩形）算出来对应哪个尺度（P2~P5中的哪一个）的。"""

    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):  # 循环是从i=0,level=2，循环到i=4,level=6，i和level是一一对应的。
        ix = tf.where(tf.equal(roi_level, level))
        # 上句，从roi_level矩阵中选出来值是level（即tf.where()的输入为True）的那些点（每个点就对应某张图中的某个外接矩形）。
        # i=0,level=2的时候，执行完了ix就是<tf.Tensor 'mask_rcnn_model/Where_10:0' shape=(?, 2) dtype=int64>
        # 【基础】二维矩阵中的位置信息
        # 输出shape=(?, 2)，?应该表示尺度是level的roi的个数（现在既不是批次中的图张数，也不是一张图中的样例数），2是因为那些点的要用2个数来表示坐标。
        #     即，输出有?行，表示现在不知道多少个roi（样例，即正例或负例）的尺度是level；
        #     每一行有两列，第一列的每一个数表示在哪张图上、第二列的数表示是这张图上的哪个样例，见《TensorFlow基础tf.where.docx》。
        level_boxes = tf.gather_nd(rois, ix)
        # 上句，（在所有的4张图中）找到当前level的所有样例的角点坐标。应该是先在rois的第0维（4即batch_size张图）里选出来ix[:,0]对应的那?个，
        #     再在rois的第1维（?即每张图中?个）里选出来ix[:,1]对应的那?个，
        # i=0,level=2的时候，执行完了level_boxes就是<tf.Tensor 'mask_rcnn_model/GatherNd:0' shape=(?, ?) dtype=float32>。
        #     其中shape=(?, ?)，第一个?是尺度为当前level的样例数（即ix的shape[0]的那个?），第二个?是4即角点数。
        #     也就是说level_boxes是一个?*4的矩阵，对应着当前level下的?个样例的角点坐标。详见26(gather_and_gather_nd).py。
        box_indices = tf.cast(ix[:, 0], tf.int32)
        # 上句，i=0,level=2的时候，执行完了box_indices就是<tf.Tensor 'mask_rcnn_model/Cast_50:0' shape=(?,) dtype=int32>。
        #     其中shape=(?,)，?是“尺度为当前level的正例和负例”的数量，然后这个box_indices中的每个数应该是表示哪张图上找到的这个样例。
        #     和下面tf.image.crop_and_resize里的用法是相呼应的。
        box_to_level.append(ix)
        # 上句，记录下每个外接矩形被映射到了哪个层级。把尺度=level的外接矩形的“二维位置索引”信息（在哪张图上、是第几个外接矩形）存到box_to_level里。
        level_boxes = tf.stop_gradient(level_boxes)
        # i=0,level=2的时候，执行完了level_boxes就是<tf.Tensor 'mask_rcnn_model/StopGradient:0' shape=(?, ?) dtype=float32>
        box_indices = tf.stop_gradient(box_indices)
        # i=0,level=2的时候，执行完了box_indices就是<tf.Tensor 'mask_rcnn_model/StopGradient_1:0' shape=(?,) dtype=int32>。
        pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, pool_shape, method="bilinear"))
        # 上句，先从特征图（feature_maps[i]是P2~P5中的一个）中裁剪该层级的那些样例（用level_boxes里的那些角点坐标去裁剪，一共box_indices个），
        #     然后放缩到self.pool_shape（即7*7），然后存到pooled里去。每次都存进去一个shape=(?, 7, 7, 256)的张量，
        #     其中第一个?应该是当前level的正例和负例数，然后7,7,256是因为输入特征图就是256层的，且现在裁剪缩放成了7*7的。
        # i=0,level=2的时候，执行完了pooled就是[<tf.Tensor 'mask_rcnn_model/CropAndResize_2:0' shape=(?, 7, 7, 256) dtype=float32>]，
        #     应该可以把里面的tensor给run掉看看。
        # 【基础】tf.image.crop_and_resize：
        #     第一个参数：图或特征图，表示原图或者特征图的张量（4维的），shape是(batch_size, 长, 宽, 通道数)。
        #                这儿是当前层级的特征图，即P5~P2中的一个，含有2（batch_size）张图。（注意，这里并不能选择是哪个层级，
        #                即不能选择是P5~P2中的哪一个，这个选择的工作是在for循环里做的，这儿只能选这个batch中哪一张图的P2。）
        #     第二个参数：裁剪框，表示要在某一张图中裁剪的所有区域，shape是(裁剪框的个数, 4)。
        #                这儿用level_boxes，即一大堆的样例的角点坐标（当然这些样例都是在当前level下的）。
        #     第三个参数：裁剪框的索引，其中的第i个数就是第i个裁剪框来自于第一个参数中的哪一张图（在batch_size张图中选一张），
        #                shape为(裁剪框的个数)。
        #     后面的参数略。
        #     所以说，比如说i=0的时候，level是2，那么输入第一个参数就成了P2，然后先用box_indices定下来在选哪张图的特征图，
        #         再根据level_boxes中的样例外接矩形（这些样例都是level=2的）裁剪掉，并且缩放了，就得到level=2层级的所有样例的、缩放后的特征图。
        #         然后循环过来，下一次i=1，level是3，输入第一个参数就成了P3，然后也是先用box_indices定下来在选哪张图的特征图，
        #         再根据level_boxes中的样例外接矩形（这次这些样例都是level=3的）裁剪掉，并且缩放了，就得到level=3层级的所有样例、缩放后的的特征图。
        #     输出的shape是：[batch * num_boxes, pool_height, pool_width, channels]，第一维的?是(批次中图张数*每张图中的样例数)。
        # 对应原文中这句：
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
    pooled = tf.concat(pooled, axis=0)
    """
    上面循环，是从不同层级的特征图P2~P5中，从所有样例（前序程序得到的结果，即rois）中选出相应层级的正例或负例，
        裁剪出来并且缩放到7*7*256这个大小。不同层级应该有不同个数的样例。
    循环完了之后，pooled是个list，里面有4个张量，代表不同的级别。每个张量都是shape=(?, 7, 7, 256)，?应该是各级别的正例或负例数。
        然后concat起来，4个张量拼接成一个。得到的pooled的shape=(?, 7, 7, 256)，此时的?应该是批次中所有图的、所有尺度的样例数（即4*150）。
        pooled就是所有图的、所有尺度的、裁剪缩放到7*7*256后的样例。现在，这些样例是按照尺度顺序排列的，下面把它按照批次里的图像序号排列。
    所以，就发现那个4（批次数）就没了，8（批次数）也没了。
    """
    box_to_level = tf.concat(box_to_level, axis=0)  # 4个张量拼接成一个。现在的shape=(?, 2)，即若干行2列的，?是批次中所有图的、所有尺度的样例数。然后每一行有两列，第一列的每一个数表示在哪张图上、第二列的数表示是这张图上的哪个样例
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    # 上句，box_range应该就是从0到总样例数-1这些数。box_range的shape=(?,1)，?同上。
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
    # 上句，把那个box_range拼接过来，box_to_level的shape=(?, 3)，?同上。
    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    # 上句，box_to_level[:, 0]和[:, 1]的shape都是(?,)，?代表批次中所有图的、所有尺度的样例数。
    #     其中box_to_level[:, 0]里面存的应该是某个样例“在哪张图上”，是一大堆从0~3之间的数（批次中一共4张图）
    #     box_to_level[:, 1]里面存的应该是某个样例“是该图中的第几个”，是一大堆从0~149之间的数（一张图中150个样例）
    #     把box_to_level[:, 0]*100000后，就是说第0张图上的样例，sorting_tensor的值应该是0~149之间，
    #         第1张图上的是在100000~100149之间，以此类推。
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
    # 上句，按照sorting_tensor的大小去按降序排序（见下），显然0~149就排在100000~100149之后。这就相当于是按照图的索引号重新排序了。
    # 【基础】用tf.nn.top_k可以做排序
    #     如果把那个k设成和输入向量的长度一样（就是那个tf.shape(box_to_level)[0]，即所有图的、所有尺度的样例数），
    #     而且把sorted保留为默认值True，那么就是把原来输入向量所有的值都取了出来，然后按大小排序。
    # 后面的.indices[::-1]是，因为tf.nn.top_k会输出[values,indices]，现在就取了后面的.indices，然后[::-1]是倒序排列。
    # 【基础】这个是可以试一下的，见《TensorFlow基础：如何试一下似乎不太懂的地方.docx》
    ix = tf.gather(box_to_level[:, 2], ix)
    pooled = tf.gather(pooled, ix)  # shape=(?, 7, 7, 256)
    # 以上两行，就是按照排序好了的ix，重新排列那个box_to_level，然后按照排列好了的box_to_level，重排pooled。
    """
    以上，得到的pooled是按照“在哪张图上”排列了的、所有图中的、所有尺度的、裁剪缩放到7*7*256后的样例。
    ?仍然是所有图的、所有尺度的样例数，即4*150。其实感觉现在就可以reshape了啊，如果reshape成shape为(4, 150, 7, 7, 256)的，
        这不就是把不同图中的（所有尺度、裁剪缩放后的）样例给分开了吗。。
        按照本函数的输入，那个4是已知的，而那个150是个?，那么reshape成(4, -1, ... )应该就能得到shape=(4, ?, 7, 7, 256)的，
        而不用像下句似的，在批次的维度上，加一个1啊
    """

    # Re-add the batch dimension
    pooled = tf.expand_dims(pooled, 0)  # shape=(1, ?, 7, 7, 256) 加了个第1维，批次中图的张数（诡异的做法，见上绿字）。

    others = [roi_level, box_to_level]
    return pooled, others


def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)  # 就是取log2(x)，2的对数。。

def fpn_classifier_graph(config, rois, feature_maps, num_rois, train_flag):
    """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.
        建立特征金字塔网络分类器、回归头。对应文章174中的图4、右图、灰色底色部分。文中也说输入ROI，输出类别和外接矩形。
        一个问题是，前面好像说【第五步】的rois就是外接矩形啊，怎么又来了一遍啊？
            →感觉好像是，因为【第五步】的rois是正例和负例的锚，并没有施加那个修正值呢，这儿似乎是要弄那个修正吧？？

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
            规范化坐标的提出外接矩形；用的时候是MaskMRCNN_model里的rois，即那些正例和负例的外接矩形。shape=(4,?,?)。
        feature_maps: List of feature maps from diffent layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
            不同尺度的特征图；用的时候是mrcnn_feature_maps, 即那个[P2, P3, P4, P5]
        image_shape（后来用config.IMAGE_SHAPE）: [height, width, depth]
            图像形状；用的时候是config.IMAGE_SHAPE，shape是[512 512 3]。
        pool_size（后来用config.POOL_SIZE）: The width of the square feature map generated from ROI Pooling.
            池化大小；用的时候是config.POOL_SIZE
        num_classes（后来用config.NUM_CLASSES）: number of classes, which determines the depth of the results
            类别数；用的时候是config.NUM_CLASSES，现在是7 or 4。
        num_rois: number of proposals
            提出数量；训练的时候是config.TRAIN_ROIS_PER_IMAGE，测试的时候是proposal_count。

        Returns:
            logits: [N, NUM_CLASSES] classifier logits (before softmax)
                输出的logits：shape实际上是(?,150,7 or 4)，好神奇啊，不知道为什么现在忽然就有数了。原来觉得该有数的时候，反而是一大堆的?。
                    不过无论如何，这个150就是一张图中的样例数，然后7就是这些样例的种类数。
                    想想也是，因为一张图中的样例数是一定的，而且是已经给出来了的啊。
                此外，注意这儿输出的是NUM_CLASSES，回想一下【第三步】得到的rpn_class_logits，只能预测是正例还是负例啊。
            probs: [N, NUM_CLASSES] classifier probabilities
            bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes
                输出的bbox_deltas：仍然外接矩形修正值，是要施加到提出的外接矩形上的。
        """
    x, others1 = pyramidroialign(rois, feature_maps, [config.POOL_SIZE, config.POOL_SIZE], config.IMAGE_SHAPE, name="roi_align_classifier")
    # 上句执行后，x的shape=(1, ?, 7, 7, 256)，已验证和用keras的一样。
    # 但问题是，我feature_maps明明是有批次大小的信息的，就是8（或者4，或者?），为什么要弄成1呢？
    #     →→那个1确实是诡异，前面分析那个?应该是4*150，所以，如果把第0维的那个1给squeeze掉，然后做全连接（用卷积代替），
    #       就应该得到shape=(?, 1, 1, 1024)的张量，然后把那两个1给删掉，可以得到(?, 1024)，
    #       然后再reshape成(4, ? ,1024)（这和原来那个(?, 150, 1024)是对应的）
    _, _, _, _, input_layers = x.get_shape().as_list()
    x = tf.squeeze(x, axis=0)  # shape=(?, 7, 7, 256)，?似乎是4*150。...这里输出的x要弄出来。
    x = MaskRCNN_2_ResNet.conv_layer(x, [config.POOL_SIZE, config.POOL_SIZE, input_layers, 1024], [1024],
                                     strides=[1, 1, 1, 1], padding='VALID', name="mrcnn_class_conv1")  # shape=(?, 1, 1, 1024)
    """
    上句，原来程序里是：
        执行后x就变成shape=(?, 150, 1, 1, 1024)，?是指一个批次中的图张数；150是一张图中的样例数；两个1是指长宽都变成1了；1024是指层数是1024层；
        还记得174文章中图4右边灰色底色部分，说了FPN，其中有一个7*7*256和两个1024，估计那个7*7*256就对应上句输入的x，第一个1024就是上句输出的x。
        然后那个150就是那个TRAIN_ROIS_PER_IMAGE，发现是在input_shapes.append(x_elem._keras_shape)这句话里弄出来的。然后他提到：
        In subsequent layers, there is no need for the `input_shape`，
        可能是说，前面某个地方已经说了这个输入shape有个150，所以现在就直接用了。具体先不管他。。。
    现在是想用tf的卷积层，所以把第0维的那个1删掉（见pyramidroialign那句后的注释），所以shape是(?, 1, 1, 1024)。
    """
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_class_bn1', is_training=train_flag)  # True
    # 上句，原来用的KL.TimeDistributed(BatchNorm...有个axis=3，这儿就没管它了，不知道行不行。不过ResNet那块，也是没管它。。
    # 似乎axis=3就是每一个层正则化的。。不知道tf里的是不是就默认对每一个层正则化了。。
    # 执行后，shape=(?, 1, 1, 1024)（原程序里是(?, 150, 1, 1, 1024)）
    x = MaskRCNN_2_ResNet.conv_layer(x, [1, 1, 1024, 1024], [1024],
                                     strides=[1, 1, 1, 1], padding='VALID', name="mrcnn_class_conv2")
    # 上句，发现原来keras里的卷积层的padding默认就是valid，所以这儿就也弄成valid了。
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_class_bn2', is_training=train_flag)  # True
    shared = tf.squeeze(tf.squeeze(x, axis=1), axis=1, name= "pool_squeeze")
    """
    【说明】上句和原程序不一样，原来执行后，shape=(?, 150, 1024)，现在则是shape=(?, 1024)，可以变成shape=(4, ?, 1024)或者(?, 150, 1024)
    但忽然觉得，似乎可以不在这个时候reshape啊，我记得原来的全连接层，都是shape分别为(?, 1024)和(1024, 7)的做tf.matmul，得到(?, 7)的张量，
        而，尽管tf.matmul确实可以做三维矩阵乘法（shape=(2, 4, 3)的和shape=(2, 3, 2)的matmul后得到(2, 4, 2)的），但是在现在的情况，
        因为不知道那个(2, 3, 2)中的第一个2，所以构造全连接中的权重项的时候会报错（大意就是构造的tf.Variable的shape不能是None或者-1），或者
        TypeError: Failed to convert object of type <class 'list'> to Tensor. Contents: [None, 1024, 7]. Consider casting elements to a supported type.
        所以考虑这儿先保留(?, 1024)的格式，先乘以(1024, 7)的权重并且和偏置向量加了，再reshape成(?, 150, 7)的吧。。。
    执行后，shared是 <tf.Tensor 'mask_rcnn_model/pool_squeeze:0' shape=(?, 1024) dtype=float32>。训练和测试的时候，都是同一个变量名的。
    """
    if config.USE_LISTA == True:
        with tf.variable_scope('LISTA_dict_learning'):
            dict_shape = [1024, config.IMAGE_MIN_DIM*config.projection_num]
            # 上句的config.IMAGE_MIN_DIM就是图像大小。
            D = tf.get_variable("weights", dict_shape, initializer=tf.contrib.keras.initializers.he_normal())
            layers, sparse = LISTA_manners.LISTA_network(shared, layer_num=config.layer_num, init_lam=config.init_lam, D=D)
            # 上句，相当于是加了一个config.layer_num层的LISTA单元，输出的sparse是(批大小*每张图中物体数, 投影数*图像长宽)，
            #     即(2*num_rois, config.IMAGE_MIN_DIM*config.projection_num)的。
        with tf.variable_scope('mrcnn_class_projection_inv'):
            w_fc_classes = MaskRCNN_2_ResNet.weight_variable \
                ([config.IMAGE_MIN_DIM * config.projection_num, config.projection_num * config.NUM_CLASSES])
            b_fc_classes = MaskRCNN_2_ResNet.bias_variable([config.projection_num * config.NUM_CLASSES])  #
            h_fc_classes = tf.matmul(sparse, w_fc_classes) + b_fc_classes  #
            # 以上是弄成L个类别预测的logits，h_fc_classes的大小应该是(批大小*每张图中物体数, 投影数*类别数)
        with tf.variable_scope('mrcnn_class_logits_ensemble'):
            w_ensemble = MaskRCNN_2_ResNet.weight_variable(
                [config.projection_num * config.NUM_CLASSES, config.NUM_CLASSES])
            b_ensemble = MaskRCNN_2_ResNet.bias_variable([config.NUM_CLASSES])  #
            h_ensemble = tf.matmul(h_fc_classes, w_ensemble) + b_ensemble
            # 以上是L个类别预测的投票集成学习。h_ensemble的大小应该是(批大小*每张图中物体数, 类别数)
            mrcnn_class_logits = tf.reshape(h_ensemble, [config.BATCH_SIZE, num_rois, config.NUM_CLASSES],
                                            name='mrcnn_class_logits')
            mrcnn_probs = tf.nn.softmax(mrcnn_class_logits, name="mrcnn_class")
        with tf.variable_scope('mrcnn_bbox_projection_inv'):
            w_fc_boxes = MaskRCNN_2_ResNet.weight_variable \
                ([config.IMAGE_MIN_DIM * config.projection_num, config.projection_num * config.NUM_CLASSES * 4])
            b_fc_boxes = MaskRCNN_2_ResNet.bias_variable([config.projection_num * config.NUM_CLASSES * 4])
            h_fc_boxes = tf.matmul(sparse, w_fc_boxes) + b_fc_boxes  #
            # 以上是弄成L个外接矩形预测值，h_fc_boxes的大小应该是(批大小*每张图中物体数, 投影数*类别数*4)
        with tf.variable_scope('mrcnn_bbox_logits_ensemble'):
            w_ensemble_b = MaskRCNN_2_ResNet.weight_variable(
                [config.projection_num * config.NUM_CLASSES * 4, config.NUM_CLASSES * 4])
            b_ensemble_b = MaskRCNN_2_ResNet.bias_variable([config.NUM_CLASSES * 4])  #
            h_ensemble_b = tf.matmul(h_fc_boxes, w_ensemble_b) + b_ensemble_b
            # 以上是L个类别预测的投票集成学习。h_ensemble_b的大小应该是(批大小*每张图中物体数, 类别数*4)
            mrcnn_bbox = tf.reshape(h_ensemble_b, [config.BATCH_SIZE, num_rois, config.NUM_CLASSES, 4],
                                    name='mrcnn_bbox')
    else:
        enhanced_features = shared
        _, feature_dim = enhanced_features.get_shape().as_list()
        with tf.variable_scope('mrcnn_class_logits'):
            w_fc_classes = MaskRCNN_2_ResNet.weight_variable([feature_dim, config.NUM_CLASSES])  # 原来的
            b_fc_classes = MaskRCNN_2_ResNet.bias_variable([config.NUM_CLASSES])  # 原来的
            h_fc_classes = tf.matmul(enhanced_features, w_fc_classes) + b_fc_classes  # 原来的
            mrcnn_class_logits = tf.reshape(h_fc_classes, [-1, num_rois, config.NUM_CLASSES], name='mrcnn_class_logits')
            mrcnn_probs = tf.nn.softmax(mrcnn_class_logits, name="mrcnn_class")
            """上面2句，全连接+软最大，就得到了分类的logits和概率。这就叫“分类头”。"""
        with tf.variable_scope('mrcnn_bbox'):
            w_fc_boxes = MaskRCNN_2_ResNet.weight_variable([feature_dim, config.NUM_CLASSES * 4])
            b_fc_boxes = MaskRCNN_2_ResNet.bias_variable([config.NUM_CLASSES * 4])
            h_fc_boxes = tf.matmul(enhanced_features, w_fc_boxes) + b_fc_boxes
            mrcnn_bbox = tf.reshape(h_fc_boxes, [-1, num_rois, config.NUM_CLASSES, 4], name='mrcnn_bbox')
            """上面3句，先是全连接，然后reshape一下，就得到了外接矩形。这就是“外接矩形头”。"""
        sparse = None
        D = None


    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, sparse, shared, D

def build_fpn_mask_graph(rois, feature_maps, image_shape, pool_size, num_classes, num_rois, train_flag):
    """这个函数先不用了！保留在这儿就是为了做个对比，这个是没有共享卷积层的。见那个build_fpn_mask_graph_with_reuse函数。
    和MaskRCNN_3_RPN.py的那个有点像。。
    """
    x, _ = pyramidroialign(rois, feature_maps, [pool_size, pool_size], image_shape, name="roi_align_classifier")  # x的shape=(1, ?, 14, 14, 256)
    input_layers = x.shape[4]
    input_layers = tf.cast(input_layers, dtype=tf.int32)
    x = tf.squeeze(x, axis=0)  # shape=(?, 14, 14, 256)，?似乎是4*150。
    x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, input_layers, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name="mrcnn_mask_conv1")
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn1', is_training=train_flag)
    x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, input_layers, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name="mrcnn_mask_conv2")
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn2', is_training=train_flag)
    x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, input_layers, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name="mrcnn_mask_conv3")
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn3', is_training=train_flag)
    x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, input_layers, 256], [256], strides=[1, 1, 1, 1], padding='SAME', name="mrcnn_mask_conv4")
    x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn4', is_training=train_flag)  # shape应该是=(?, 14, 14, 256)

    x = MaskRCNN_2_ResNet.deconv_layer1(x, 256, 2, strides=2, name='mrcnn_mask_deconv_pre')  # shape应该是=(?, 28, 28, 256) dtype=float32
    x = tf.nn.relu(x, name='mrcnn_mask_deconv')
    x = MaskRCNN_2_ResNet.conv_layer(x, [1, 1, 256, num_classes], [num_classes], strides=[1, 1, 1, 1], padding='SAME', name="mrcnn_mask_pre")
    x = tf.sigmoid(x, name="mrcnn_mask_pre1")  # shape=(?, 28, 28, 7)已验证
    mrcnn_mask = tf.reshape(x, [-1, num_rois, 28, 28, num_classes], name='mrcnn_mask')  # shape=(?, 150, 28, 28, 7) dtype=float32已验证
    return mrcnn_mask

def build_fpn_mask_graph_with_reuse(rois, feature_maps, image_shape, pool_size, num_classes, num_rois, train_flag):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    建立FPN网络的掩膜头的计算图。对应文章174中的图4、右图、白色底色部分（但不完全一样）。
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    以上输入和上面那个函数一样的。

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    x, _ = pyramidroialign(rois, feature_maps, [pool_size, pool_size], image_shape, name="roi_align_classifier")  # x的shape=(1, ?, 14, 14, 256)
    input_layers = x.get_shape().as_list()  # 注意此处必须用x.get_shape().as_list()，才能让shape变成5个int组成的数组，供后面使用的。结果是[1, None, 14, 14, 256]。
    depth = input_layers[4]  # 256
    x = tf.squeeze(x, axis=0)  # shape=(?, 14, 14, 256)，?似乎是4*150。
    with tf.variable_scope("mrcnn_mask_conv1"):  # 就仍然用原来的那个name作为这个的variable_scope。
        x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, depth, 256], [256], name="m1", strides=[1, 1, 1, 1], padding='SAME')
        x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn1', is_training=train_flag)  # 这个不确定是不是要放在tf.variable_scope里面。
    with tf.variable_scope("mrcnn_mask_conv2"):
        x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, 256, 256], [256], name="m2", strides=[1, 1, 1, 1], padding='SAME')
        x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn2', is_training=train_flag)
    with tf.variable_scope("mrcnn_mask_conv3"):
        x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, 256, 256], [256], name="m3", strides=[1, 1, 1, 1], padding='SAME')
        x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn3', is_training=train_flag)
    with tf.variable_scope("mrcnn_mask_conv4"):
        x = MaskRCNN_2_ResNet.conv_layer(x, [3, 3, 256, 256], [256], name="m4", strides=[1, 1, 1, 1], padding='SAME')
        x = MaskRCNN_2_ResNet.batch_normalization(x, scope='mrcnn_mask_bn4', is_training=train_flag)  # shape应该是=(?, 14, 14, 256)

    with tf.variable_scope("mrcnn_mask_deconv"):
        output_shape_0 = tf.shape(x)[0]
        x = MaskRCNN_2_ResNet.deconv_layer_with_reuse(x, [2, 2, 256, 256], [256],
                              [output_shape_0, 28, 28, 256], strides=[1, 2, 2, 1], padding='SAME')
        # 上句，输出shape应该是=(?, 28, 28, 256) dtype=float32。验证了确实是。
        x = tf.nn.relu(x, name='mrcnn_mask_deconv')
    with tf.variable_scope("mrcnn_mask_pre"):
        x = MaskRCNN_2_ResNet.conv_layer(x, [1, 1, 256, num_classes], [num_classes], name="mrcnn_mask_pre", strides=[1, 1, 1, 1], padding='SAME')
        x = tf.sigmoid(x, name="mrcnn_mask_pre1")  # shape=(?, 28, 28, 7)已验证
    mrcnn_mask = tf.reshape(x, [-1, num_rois, 28, 28, num_classes], name='mrcnn_mask')  # shape=(?, 150, 28, 28, 7) dtype=float32已验证
    return mrcnn_mask