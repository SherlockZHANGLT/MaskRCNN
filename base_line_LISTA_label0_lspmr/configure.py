#coding=utf-8
"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    TRAIN_STEP = 40000  # 训练步数 20000
    TRAIN_STEP_MP = 3000  # 训练步数 5400，后来发现如果加上路径损失，训练3000步就够了。

    FEN_network = "Resnet101"
    assert FEN_network in ["Resnet101", "Resnet50", "VGG19", "VGG19_FCN", "VGG19_FCN_simp", "Googlenet", "Densenet"]
    # Resnet101、Resnet50、VGG19、VGG19_FCN、Googlenet、Densenet

    lspmr_weight = 0.0005

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "organs"  # Override in sub-classes  原来写的是None。

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000 # 应该是1000，调试的时候可以减少，节约时间。

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]  # 似乎要和RPN_ANCHOR_SCALES里的个数一样，见后。

    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 9  # 探测数据集的，是背景 + 9种脊骨。（PS：原来分割数据集的，背景 + 6种器官；滑脱的是1+3种器官。）

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # 正方形的锚的边长，不同长宽比的锚的边长会在此基础上乘或除一个数。
    # PS：原来是(32, 64, 128, 256, 512)，现在因为原图就是512*512的，椎骨不可能跟原图一边大，所以锚边长缩小了。
    # 【注意】这个玩意的个数，可能和BACKBONE_STRIDES里的个数要相等，否则会报错。
    #     也就是说，一个RPN_ANCHOR_SCALES（锚边长）对应一个BACKBONE_STRIDES（多少个像素点有一个锚中心点）

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride 锚步长
    # If 1 then anchors are created for each cell in the backbone feature map. 如果步长是1，就在骨架特征图的每个元胞上创建锚。
    # If 2, then anchors are created for every other cell, and so on. 如果步长是2，就是每隔一个元胞创建一个锚。
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.  用来过滤“RPN提出来的区域”的“非最大化抑制”的阈值。
    # You can reduce this during training to generate more propsals.  这个阈值越小，就能生成更多的提出。
    RPN_NMS_THRESHOLD = 0.5  # testing的时候似乎是0.6，原来给的是0.7。
    RPN_MIN_CONFIDENCE_MP = 0.9  # 这个是测试的时候，认为RPN分值大于0.9的，可能是正例。

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 200  # 原来给的是2000，在180820版程序里试过改成200，发现训练和测试都没有报错
    POST_NMS_ROIS_INFERENCE = 100  # 原来给的是1000，在180820版程序里试过改成100，发现训练和测试都没有报错

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 512  # 原来给的是800
    IMAGE_MAX_DIM = 512  # 原来给的是1024，但感觉显然不必要，因为我的图都是512*512的啊。弄成1024反而不好，而且会OOM。
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  # 这个是在MRCNN_utils.py里的那个mold_image函数里用的！
    # 只在测试的时候用了，训练的时候，好像注释掉这句也没事。。但，其实也没用到那个函数输出的结果，而只是用了它的形状。

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.15
    # 上句的参数，在DETECTION_MIN_CONFIDENCE固定为0.98的情况下，0.33会多弄出来提出  0.1会少弄出来提出  0.12和0.1差不多但还不如0.1

    # Number of ROIs per image to feed to classifier/mask heads  传递给“分类/掩膜头”的每张图片上的ROI数
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # 传递给“分类/掩膜头”的每张图片上的ROI数。掩膜RCNN的论文用了512，但RPN网络通常不能生成足够的正例提出，以满足需求，并保证正负例的比例是1：3。
    # 你可以通过调整RPN网络的RPN_NMS_THRESHOLD，来增加提出的数量（应该就是那个阈值小了，提出数就会变多，然后正例提出就会多）。
    """其实感觉这个值还是应该和正例数、正负例比例结合起来。见下。"""
    ONE_INSTANCE_EACH_CLASS = True
    if ONE_INSTANCE_EACH_CLASS:
        TRAIN_ROIS_PER_IMAGE = int((NUM_CLASSES - 1) / ROI_POSITIVE_RATIO)
        # 如果每一类都只有1个物体，那么，正例数就是(NUM_CLASSES - 1)，所以每张图的ROI数应该就是上式。
    else:
        TRAIN_ROIS_PER_IMAGE = 20  # 传递给“分类/掩膜头”的每张图片上的ROI数，原来是200；曾用过150效果不错，然后调成50试了试同一张图，
        # 发现：似乎用150的时候，最后得到的信心分值会高一些。（不过我不明白的是，为啥这个随便改，而可以不报错啊？训练的时候，我可是用150训练的啊。。）
        # 20也用过，还不错。现在觉得应该是。

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100  # 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.95  # 训练MRCNN的时候，用0.95。
    DETECTION_MIN_CONFIDENCE_MP = 0.9  # 训练信息传递的时候，用0.9。
    """上句的参数，据调试经验，应该是要随着一些别的参数而改动的。
    如，滑脱数据集中，TRAIN_ROIS_PER_IMAGE=20、POST_NMS_ROIS_INFERENCE=100的时候，
        就会发现正确的脊椎骨都是0.99甚至0.999的，而好多不对的都有0.98呢，所以这个阈值可以高一些。
    但是呢，如果TRAIN_ROIS_PER_IMAGE=50、POST_NMS_ROIS_INFERENCE=100的时候，
        就会发现好多正确的脊椎骨都只有0.8或者0.7，这个时候这个阈值就要低一些。
    """

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    MP_NMS_THRESHOLD_ONE = 0.25
    MP_NMS_THRESHOLD_TWO = 0.25
    MP_ignore_first_label = True

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001  # 原来是0.001
    LEARNING_RATE_MP = 0.06  # 信息传递的学习率 原来CRF是0.01，不过发现那个损失有4.几，所以索性用大一些试试吧
    end_learning_rate = 0.00001
    LEARNING_MOMENTUM = 0.9
    num_epochs_per_decay = 10 # 10个时代就降低一次学习率
    learning_rate_decay_type = 'exponential'  # 学习率指数衰减
    learning_rate_decay_factor = 0.96
    decay_steps_MP = 1000  # 1000步衰减一次学习率
    learning_rate_decay_factor_MP = 0.96

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001  # 原来是0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    """USE_RPN_ROIS：训练的时候，用RPN的ROI还是外部生成的ROI来训练
    如果希望用程序生成的ROI来训练头，而不是用RPN出来的ROI来训练头，就把这个设成False。
    """
    USE_RPN_ROIS = True

    # 以下是LISTA用到的参数
    USE_LISTA = True
    weights_sparsity = 1  # 是为了让稀疏损失和其他损失在开始的时候，在同一个数量级上。
    weights_dense = 1  # 是为了让稀疏损失和其他损失在开始的时候，在同一个数量级上。另外，按照094文章的论文里，
    # 这个浓密的权重大概是稀疏的1/1.7，但是如果直接用0.05/1.7=0.029的话，又太小了，好像乘上100才差不多，所以索性就用1好了。
    projection_num = 4  # 4个的话，就是在0 45 90 135这四个角度去投影，应该差不多就够了吧，那么多也似乎没啥用。。
    layer_num = 3
    init_lam = 0.1

    epochs_LISTA = 5  # 训练多少个时代的LISTA？

    # 以下是字典学习用到的参数
    use_dict = False
    if use_dict:
        compress_size = 800
        k_value = 500  # 60
        # assert compress_size / NUM_CLASSES >= k_value, '字典列数compress_size除以类别数，必须大于非零个数。'
        # PS，而且还必须大一些，否则可能效果不好，比如说k_value是25、compress_size是25，就不怎么好。最好把compress_size弄大一些吧。
        # 不过后来发现，如果不用他那个KSVD去初始化，似乎也没啥事儿。。
        dict_weights = "exponential"
        assert dict_weights in ["None", "fixed", "less_than_main", "exponential", "linear"]
        dict_weights_recon_ini = 0.1  # 原来用的是0.1，改成0.0就是不要字典了。
        dict_weights_recon_decay_factor = 0.7
        dict_weights_consi_ini = 1.0  # 原来用的是1.0，改成0.0就是不要字典了。
        dict_weights_consi_decay_factor = 0.7
        dict_weights_consi_ini1 = 0.0  # 类似于LSPMR损失的那个
        dict_weights_consi_decay_factor1 = 0.7
    else:
        pass  # 不定义字典学习的参数了。

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size 从输入图像大小计算骨架大小
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])
        # self.BACKBONE_STRIDES里是[4 8 16 32 64]这五种，然后self.BACKBONE_SHAPES就是[[128 128]\n [64 64]\n ... [8 8]]。

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class Config_spondy(object):
    """
    滑脱分级网络的一些训练参数
    """
    ini_learning_rate = 0.0001
    num_epochs_per_decay = 4
    learning_rate_decay_type = 'exponential'
    end_learning_rate = 0.00000001
    learning_rate_decay_factor = 0.96
    learning_momentum = 0.96
    l2_weight = 0.00001
    num_gradings = 4

    # 这个是不是就不需要用__init__了？