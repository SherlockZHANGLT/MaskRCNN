#coding=utf-8
import numpy as np
import logging
import MRCNN_utils
import LISTA_manners
# import configure  # 要留着，测试的时候用了
# import  make_dataset  # 要留着，测试的时候用了

############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        输入的尺度scales参数：到此处是一个数，如8。正方形锚的边长，单位是像素点。
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        输入的比例ratios参数：到此处是一个向量，[0.5,1,2]。锚的长宽比。
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
        输入的形状shape参数：到此处是一个向量，[128 128]，骨架大小。
    feature_stride: Stride of the feature map relative to the image in pixels.
        输入的特征步长参数：到此处是一个数，如4。FPN金字塔结构中的每个层的步长。
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
        输入的锚步长参数：一直是1。

    然后返回的是boxes，是角点坐标。
    """
    # Get all combinations of scales and ratios 枚举尺度和比例的各种组合
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()  # 如果输入的scales是8，这里就是[8 8 8]
    ratios = ratios.flatten()  # 仍然是[0.5,1,2]

    # Enumerate heights and widths from scales and ratios  该尺度（边长为8个像素点的尺度）下，不同长宽比的情况下的长宽。
    heights = scales / np.sqrt(ratios)  # array([11.3137085, 8., 5.65685425])
    widths = scales * np.sqrt(ratios)  # array([5.65685425, 8., 11.3137085 ])

    # Enumerate shifts in feature space  枚举特征空间内的移动   那个np.arange(...)是0~127之间的数，然后乘以feature_stride（4）。
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride  # array([0, 4, 8, 12, ..., 504, 508])。
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride  # 和上面一样，幸好这里不是tensor，否则只能看到shape就傻了。。
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)  # shifts_x, shifts_y都变成128*128的矩阵，跟MATLAB里差不多。
    """
    【重要】那个feature_stride，物理意义上讲，就是相邻的两个锚的中心位置的距离。比如说，第一次执行这个函数的时候feature_stride是4，
        那么，两个相邻的锚中心点的距离就是4。比如说，一个锚的中心点是(0,0)，则它右边的锚的中心点是(4,0)，下标的中心点是(0,4)。
        当然，以(0,0)/(0,4)/(4,0)为中心点的锚都有好几个（长宽比不同），但他们的中心点位置都是这样给定的。
    这样的话，feature_stride需要和那个shape（特征空间里的大小）对应上，比如说，如果feature_stride就是4，shape就是[128 128]，
        这样，二者一乘就成了原图大小512，即把特征图空间里的矩形框弄回了原图中去。
        然后下次执行这个函数，feature_stride成了8，而shape成了[64 64]。
    另外，那个锚边长，即输入的scales，和这个feature_stride和shape是没关系的，我现在只需要保证feature_stride和shape的乘积是512，
        而，锚边长是可以随便取的（事实上，一共取了8, 16, 32, 64, 128这五种。
        本函数中的注释基本上是以8为例的，所以那个heights和widths是[11.3137085, 8., 5.65685425]什么的。）
    """

    # Enumerate combinations of shifts, widths, and heights  枚举移动、宽、高的组合（好像就是外接矩形中心x、y坐标、长度和宽度的所有组合）
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    # 上句，box_widths就是16384个[5.65685425, 8., 11.3137085]，
    #     box_centers_x就是[0,0,0]...[508,508,508]这128个向量重复128遍，得到16384个向量。
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)  变形，得到一个(y,x)的列表box_centers、和一个(h,w)的列表box_sizes。
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    # 上句，np.stack(...)是先把外接矩形中心y坐标和x坐标拼起来得到(16384, 3, 2)的张量，然后reshape变成(49152, 2)的张量。
    #     看一下就知道了，是[0,0]，[0,0]，[0,0]，[0,4]，[0,4]，[0,4]...[0,508]，[0,508]，[0,508]，[4,0]，[4,0]，[4,0]，[4,4]，[4,4]，[4,4]，...这样的，
    #     相当于是每个中心点位置写了3遍，应该是为了对应一会儿的3种长宽比的。
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    """
    上句，就是[11.3137085,5.65685425]，[8,8],[5.65685425,11.3137085]重复16384遍，得到49152个向量。
        然后就可以发现，box_centers和box_sizes这两个矩阵中的每一行，恰好就是一种中心点坐标和长宽的组合。
        所以说这两个矩阵就列举了所有的区域提出中心点坐标和长宽的组合。
    【重要】所以这就应该知道这个“锚”（不同分辨率下、一些不同形状、位置、大小的矩形框）是怎弄出来的了（其实就是先放缩再枚举）。
    """
    # Convert to corner coordinates (y1, x1, y2, x2)  转化成角点坐标
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # 上句，先把上面两个矩阵（所有中心点坐标和长宽的组合）拼接起来，然后把长宽变成角点坐标（注意角点坐标不是中心点坐标！）。
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    在特征金字塔的不同层级生成锚。每个尺度scale和金字塔的一个层级有关，但比例ratio是用在金字塔所有层级上，
        具体是说，下面的那个for循环，每一次循环都是scales[i]，但是却用了整个ratio，这就是只用了一个尺度scale，而用了所有的比例ratio。

    输入参数（同model.py中utils.generate_pyramid_anchors函数）：
    scales：正方形锚的边长，单位是像素点。这里取了(8, 16, 32, 64, 128)这五种。
    ratios：锚的长宽比，现在只有0.5/1/2这三种。
    feature_shapes：从输入图像大小计算骨架大小，目前是个矩阵，[[128 128]\n [64 64]\n ... [8 8]]，似乎是五种大小的骨架图
    feature_strides：FPN金字塔结构中的每个层的步长，[4, 8, 16, 32, 64]这五种。
    anchor_stride：如果步长是1（现在确实就是1），就在骨架特征图的每个元胞上创建锚（organs_training.py或config.py都有）。

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    返回锚：锚是个五个元素的array。按照给定的尺度顺序排序，所以，第一个是scale[0]的锚，第二个是scale[1]的锚，。。。
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):  # 对每个尺度（scales），用generate_anchors生成锚，然后拼接到anchors里去。
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],  # generate_anchors见上。
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

def get_image_ids_for_next_batch(dataset, batch_size, shuffle=True):
    """【基础】按照索引号，取出一个批次的图像。
    注意这个函数的写法，很有用！！主要是要设定dataset的三个成员变量：_index_in_epoch、_epochs_completed、_image_ids_shuffle，
        分别代表当前时代下，弄到了哪张图；当前弄到了第几个时代；当前时代下的乱序图片序号序列！
    """
    num_examples = len(dataset.image_ids)  # 数据集里的总元素个数。
    start_id = dataset._index_in_epoch  # 这个批次，从数据集里的第几个元素开始。
    image_id = np.copy(dataset.image_ids)  # 0~19的array，相当于range(num_examples)。
    # 处理第一个时代，获得打乱了顺序的“图像信息序列”。
    if dataset._epochs_completed == 0 and start_id == 0 and shuffle:  # _epochs_completed似乎应该在外面函数里设计。。
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        dataset._image_ids_shuffle = image_id[perm0]  # 数据集里所有图像，按照打乱了的顺序，排成一个list。
    if start_id + batch_size <= num_examples:  # 在一个时代内取数据
        dataset._index_in_epoch += batch_size
        end_id = dataset._index_in_epoch
        image_id_selected = dataset._image_ids_shuffle[start_id:end_id]
    else:
        dataset._epochs_completed += 1  # 这个时代结束
        # 把数据集中剩下的数据弄出来
        rest_num_examples = num_examples - start_id
        image_id_rest_part = dataset._image_ids_shuffle[start_id:num_examples]
        # Shuffle the data
        if shuffle:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)  # 如果shuffle，重新弄一组打乱了的数据。
            dataset._image_ids_shuffle = dataset._image_ids_shuffle[perm]
        # 开始新时代
        start_id = 0
        dataset._index_in_epoch = batch_size - rest_num_examples
        end_id = dataset._index_in_epoch
        image_id_new_part = dataset._image_ids_shuffle[start_id:end_id]
        image_id_selected = np.concatenate((image_id_rest_part, image_id_new_part), axis=0)
    """以上if-else是得到要取的那几张图在输入数据集的序号，下面取出相应的图像索引号，即在“新旧自编号对照表”中是哪个病人。"""
    dataset_patiend_ids = []
    for i in image_id_selected:
        patiend_id_this = dataset.image_info[i]['patient']
        dataset_patiend_ids.append(patiend_id_this)
    dataset_patiend_ids = np.array(dataset_patiend_ids)
    return image_id_selected, dataset_patiend_ids


def get_inputs_for_MaskRCNN(dataset, config, shuffle=True, augment=True, batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    数据生成器：返回图像和相应的目标类别ID（就是金标准类别呗）、外接矩形修正值、掩膜。

    dataset: The Dataset object to pick data from
        输入dataset：从这个输入里选取数据。调用的时候，输入的是train_dataset或val_dataset，应该是从这两个数据集中选取。
    config: The model config object
        输入config：训练设置参数
    shuffle: If True, shuffles the samples before every epoch
        输入shuffle：在每个时代钱是否打乱样本次序。
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
        输入augment：是否给图像弄图像增强。
    batch_size: How many images to return in each call
        输入batch_size：每次返回多少张图。（就是一个批次里的图像张数）


    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    输出一个Python生成器，生成两个list，其中inputs就是下面那7项
        （看了一下，正好和后面class MRCNN里最后的那个inputs里的7项一样，说明这个就是生成那个KL.Model的输入的）
        然后outputs在训练的时候一般是空的，但如果detection_targets是True，会生成生成探测目标（类别索引、外接矩形、掩膜）。
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        特别说一下这个rpn_match：在本函数中，就是那个batch_rpn_match，一开始初始化为(8, 65472, 1)的0矩阵，然后去填充的。
        那么，就说明那个N就是锚点个数（没做选择、细化之前的锚点个数）了啊。
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        也特别说一下这个rpn_bbox：首先，它是锚外接矩形的修正值，而不是外接矩形本身！
        然后，在本函数中是batch_rpn_bbox，一开始初始化为(8, 256, 4)的0矩阵，然后去填充的。
        注意到，这个N和上面的rpn_match中的N是不一样的啊，这儿是训练用的锚数，上面是总的锚数。。
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        在本函数中是batch_gt_class_ids，一开始初始化为(8, 100)的0矩阵。
        也就是说，这个MAX_GT_INSTANCES和前面的两个N都不一样的。
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    # 上句，dataset.image_ids是从0到dataset中的图像张数-1，
    #     然后这个函数里return的self._image_ids又是在prepare函数里给赋值的。现在这个image_ids是从0到98，共99张。
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                         config.RPN_ANCHOR_RATIOS,
                                         config.BACKBONE_SHAPES,
                                         config.BACKBONE_STRIDES,
                                         config.RPN_ANCHOR_STRIDE)
    # 上面生成锚。和class MaskRCNN()里第二步生成锚的那句一样。 shape=(65472, 4)。这个东西输出的anchors.dtype是dtype('float64')。
    anchors = anchors.astype('float32')  # 变成float32类型，因为后面MaskRCNN_4_Proposals里的一些东西需要float32。

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)  # 求余数，如3%99=3、102%99=3。
            if shuffle and image_index == 0:  # 一开始的时候，image_index=-1，上句+1后就是0，所以一开始的时候会把image_ids打乱。
                np.random.shuffle(image_ids)  # image_ids原来是0~98顺序拍好，现在是乱序的。
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]  # 乱序的image_index里的第0,1,2,3...个。现在的image_id就代表，当前想处理的图，是dataset中的第几张图。
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, _ = \
                MRCNN_utils.load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
            """
            以上，随机选出来一张图，用load_image_gt函数弄出来图像、图像meta、金标准类别、外接矩形、掩膜。随机取出来一张图，image_meta这次看到了，是：
                array([ 61, 512, 512,   3,   0,   0, 512, 512,   1,   1,   1,   1,   1,   1,   1])，
                然后看了一下，第一个61是图像的索引号，第二到四个512,512,3是图像大小，第五到八个0,0,512,512是windows大小，后面那七个1是类别序号，
                    似乎是表示7个类别都有，但我不明白的是，为什么有了金标准类别gt_class_ids，还要有那个类别序号？？？
            """

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
            """
            上面是说，如果某张图没有物体，就pass掉。某张图是否有物体，是用gt_class_ids中是否有>0的值判断的。
                也就是说，gt_class_ids>0就表示某种物体。
            """

            # RPN Targets 可能是在弄RPN网络的金标准。class MaskRCNN里的五个loss，前两个就是用input_rpn_match和input_rpn_bbox作为金标准。
            rpn_match, rpn_bbox = MRCNN_utils.build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)
            """
            上句，和class MaskRCNN里用的build_rpn_model（在【从图片到提出的第三步】里用了）名字有点像，输出也差不多，
                但是二者的输入完全就不一样，第三步是没有用锚的，从特征直接弄出来rpn_class和rpn_bbox；
                而这里用了锚，而且还弄了正例负例什么的，倒更像是【从图片到提出的第五步】，详见build_rpn_targets里的注释。
            然后输出的是：
                rpn_match（锚和金标准外接矩形之间的匹配关系）：
                    这东西大小和锚的数量相等，里面一个元素都是1或者-1或者0，表示正例负例和中性例。
                rpn_bbox（外接矩形修正值）：
                    这东西不是锚的那个框（理解为外接矩形吧），也不是金标准外接矩形，而是二者的差异。
            这两个输出，就是data_generator函数的七个输出中的两个。然后那个class MaskRCNN用它们作为金标准。
            """

            # Init batch arrays
            if b == 0:  # b是一个批次里的每一张图。
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)  # 输出是个(8, 15)的0矩阵。那个batch_size=8，image_meta.shape是(15,)。
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)  # (8, 65472, 1)的0矩阵
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)  # (8, 256, 4)的0矩阵
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)  #  (8, 512, 512, 3)的0矩阵
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)  # (8, 100)的0矩阵
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)  # (8, 100, 4)的0矩阵
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))  # (8, 56, 56, 100)的0矩阵
                else:
                    batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:  # 如果金标准物体数太多，做个下采样
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                print(gt_class_ids)
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch  以下，把这张图的相应数据加入到各个array中。
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = MRCNN_utils.mold_image(image.astype(np.float32), config)  # 是不是这个鬼玩意把图像弄得反而不对了？？好像原来图片都是0~255的啊，现在是个什么鬼。。
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            b += 1  # 弄完了这张图的所有金标准（指batch_image_meta、batch_gt_class_ids等等）之后，开始弄下一张图。此时需要重新执行load_image_gt等函数。

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                return anchors, inputs, outputs

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def get_batch_inputs_for_MaskRCNN(dataset, image_ids, anchors, config, num_rois, augment=False):
    class_label_pos = [28, 85, 142, 199, 256, 313, 370, 427, 484]
    batch_images = []
    batch_gt_mask = []
    batch_image_meta = []
    batch_gt_class_ids = []
    batch_gt_boxes = []
    batch_gt_masks = []
    batch_rpn_match = []
    batch_rpn_bbox = []
    batch_sparse_radon = np.zeros([config.BATCH_SIZE, num_rois,config.IMAGE_SHAPE[0]*config.projection_num])
    for co, image_id in enumerate(image_ids):
        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_mask = \
            MRCNN_utils.load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
        """注意，上句load_image_gt中，调用了load_image函数，就把灰度图变成了RGB图了。"""
        # gt_mask是这一张图的、语义分割掩膜（用来弄那个GAN的）。注意和那个gt_masks不一样。
        if not np.any(gt_class_ids > 0):
            print("警告：在第%d张图的金标准中，没有看到任何正例！" % image_id)
            continue
        rpn_match, rpn_bbox = MRCNN_utils.build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)
        sparse_radon = np.zeros([num_rois, config.IMAGE_SHAPE[0]*config.projection_num])  # 单张图的稀疏表示，应该是有
        for i in range(gt_boxes.shape[0]):
            gt_box = np.expand_dims(gt_boxes[i, :], axis=0)
            sparse_radon0, _ = LISTA_manners.get_sparse_gt(dataset.image_info[0]['height'], dataset.image_info[0]['width'],
                                                           gt_box, projection_num=config.projection_num)
            # 上句，radon变换，构造每一个物体的稀疏表示金标准（每个物体往4个投影轴上的投影），是(config.IMAGE_SHAPE[0](512), config.projection_num(4))的

            for j in range(config.projection_num):
                sparse_radon0[class_label_pos[9 - gt_class_ids[i]], j] = 1
            #以上，给类别标签到稀疏编码

            sparse_radon1 = np.reshape(sparse_radon0, [1, config.IMAGE_SHAPE[0] * config.projection_num])
            # 上句，因为Python的radon变换输出的是“原图大小*投影直线数”的，所以上句要展平成这样的shape。这个是没关系的，仍然是所有的投影结果共同监督的。
            sparse_radon[i, :] = sparse_radon1
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:  # 如果金标准物体数太多，做个下采样
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            print(gt_class_ids)
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
        batch_images.append(image)
        batch_gt_mask.append(gt_mask)
        batch_image_meta.append(image_meta)
        gt_class_ids_padded_zeros = np.zeros((config.MAX_GT_INSTANCES), dtype=np.int32)
        # 上句，因为每张图gt_class_ids的个数未必相同，所以直接append可能报错，所以先补齐了0。
        gt_class_ids_padded_zeros[:gt_class_ids.shape[0]] = gt_class_ids  # 前面若干个0替换成gt_class_ids中的类别标签。
        batch_gt_class_ids.append(gt_class_ids_padded_zeros)
        gt_boxes_padded_zeros = np.zeros((config.MAX_GT_INSTANCES, 4), dtype=np.int32)  # 类似于上面
        gt_boxes_padded_zeros[:gt_boxes.shape[0]] = gt_boxes
        batch_gt_boxes.append(gt_boxes_padded_zeros)
        if config.USE_MINI_MASK:
            batch_gt_masks_padded_zeros = np.zeros((config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], config.MAX_GT_INSTANCES))  #
        else:
            batch_gt_masks_padded_zeros = np.zeros((image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
        batch_gt_masks_padded_zeros[:,:,:gt_masks.shape[-1]] = gt_masks
        batch_gt_masks.append(batch_gt_masks_padded_zeros)
        batch_rpn_match.append(rpn_match)
        batch_rpn_bbox.append(rpn_bbox)
        batch_sparse_radon[co, :, :] = sparse_radon
    batch_images = np.stack(batch_images, axis=0)  # (batch_size, 512, 512, 3)
    batch_image_meta = np.stack(batch_image_meta, axis=0)  # (batch_size, 15)
    batch_gt_mask = np.stack(batch_gt_mask, axis=0)  #  应该是(batch_size, 512, 512)
    batch_gt_class_ids = np.stack(batch_gt_class_ids, axis=0)  # (batch_size, 100)
    batch_gt_boxes = np.stack(batch_gt_boxes, axis=0)  # (batch_size, 100, 4)
    batch_gt_masks = np.stack(batch_gt_masks, axis=0)  # (batch_size, 56, 56, 100)
    batch_rpn_match = np.stack(batch_rpn_match, axis=0)  # (batch_size, 65472)
    batch_rpn_match = np.reshape(batch_rpn_match, [batch_rpn_match.shape[0], batch_rpn_match.shape[1], 1])  # 升维变成(batch_size, 65472, 1)
    batch_rpn_bbox = np.stack(batch_rpn_bbox, axis=0)  # (batch_size, 256, 4)
    batch_sparse_radon = np.stack(batch_sparse_radon, axis=0)  # (batch_size, num_rois, 图像大小*投影数)
    inputs_for_MRCNN_dict_feeding = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_gt_mask, batch_sparse_radon]
    return inputs_for_MRCNN_dict_feeding
########################################################################################################################
#          以下是测试本函数的部分，即，在这个函数里直接执行get_inputs_for_MaskRCNN函数，得到MaskRCNN模型的金标准输入。。         #
#          比较好的是，别的函数调用这个.py文件里的函数时，下面这些测试用的东西也不用删掉。。                                     #
#          不过，如果是要import这个.py文件的话，最好还是把下面的东西注释掉，否则这些会跑一遍。。。                               #
########################################################################################################################

# config = configure.Config()  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# Fold = 5  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# dataset_dir = './SPINEtfrecords/'  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# dataset_name = 'spine_segmentation'  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# dataset_split_name_test = 'test_%s_fold'%Fold  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# batch_size_test = 1  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# num_readers_test = 4  # 试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# dataset = make_dataset.convert_SPINEtfrecord_to_HANdata(dataset_dir, dataset_name, dataset_split_name_test, batch_size_test, num_readers_test)
# # 上句，试着直接执行一下get_inputs_for_MaskRCNN函数，这儿是在做准备。。
# anc, inp, out=get_inputs_for_MaskRCNN(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)  # 试着直接执行一下get_inputs_for_MaskRCNN函数
# print(inp)