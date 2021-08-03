#coding=utf-8
"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

#import sys
#import os
#import math
import random
import matplotlib
import copy
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.ndimage
# import matplotlib.pyplot as plt


############################################################
#                以下是训练时用到的其他函数                   #
############################################################

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    保留长宽比，缩放图像
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
        输入的padding：如果这个是True，就补零，让图像的大小是max_dim*max_dim

    Returns:
    image: the resized image
        返回的image：放缩了的图像。
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
        返回的图像窗口：如果提供了max_dim，可能在返回的图像中做了补零。如果这样的话，这个窗口就是补零后的图像（full image）中，
            去除了补零的地方。x2和y2不算在去除补零了的地方。
    scale: The scale factor used to resize the image
        返回的scales：图像缩放因子。
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        返回的padding：给图像加上的补零。
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding



def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    用给定的尺度和补零来缩放掩膜。
        通常，这个尺度和补零是从resize_image()得到的，来保证图像和掩膜被同步地缩放。
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask




def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    从掩膜计算外接矩形。
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]  # 掩膜值为1的地方。随机用一张图验证过，发现np.any(m, axis=0)是一大堆TRUE和FALSE，然后np.where(...axis=0)就是从左到右一列列地去找，找到m中有1的那些列（此例的输出为232,233,234...271），后面的[0]是个小细节，把(?,)变成()。
        vertical_indicies = np.where(np.any(m, axis=1))[0]  # 类似上上句，从上到下一行行地找，找到m中有1的那些行。
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to resizing or cropping. Set bbox to zeros
            # 如果没有掩膜，就把外接矩形全都置零。
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)



def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou



def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps



def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_mask()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask



def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask



def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use parse_image_meta() to parse the values back.
    取出图像的属性，放到1个1D向量中去。（下面那个parse_image_meta是读取这些图像属性的。）

    image_id: An int ID of the image. Useful for debugging.
        输入image_id：图像序号
    image_shape: [height, width, channels]
        输入image_shape：图像的形状
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
        输入window：补零后，原有图像的位置
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
        输入active_class_ids：图像所在的数据集中的、可用的class_ids的列表。
            如果要用多个数据集中的图像训练，而这些数据集中并没有所有的类别的时候，这就是有用的。
            好像是说，如果第一个数据集有7种类别，第二个数据集只有其中的5种，那么第二个数据集中取出来的图像，
            active_class_ids中就有两个数是0吧。
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta



def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    加载一张图片的金标准数据：图像、掩膜、外接矩形。
    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
        输入augment：是否进行数据增强
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.
        输入use_mini_mask：是否局部放大掩膜。

    Returns:输出：和data_generator里的一样。
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    # 上句，应该是utils里的那个load_image函数吧，但是发现image里的东西全都是0啊（其实不是的，
    #     全是0是因为我看的是View as Array里看的是image[0]，是个512*3的矩阵，这相当于是图像第一列的三个通道，
    #     由于在图像边上，当然都是0了，但是试了试弄个image[111]看看，就有好多不是0的了啊。）。。
    masks, class_ids = dataset.load_mask(image_id)
    # 上句，应该是organs_training或testing里的那个load_image函数（utils里的那个只能生成一大堆的0好像），
    #     发现masks里的东西也都是0（然而实际上不是，m = masks[:, :, 0]再看m就可以发现有一些1的），
    #     然后class_ids是1 3 4三个数（或者别的数吧）
    """以上两句，其实可以理解为，调用函数，从dataset数据结构中解包出来原图、掩膜、金标准分类。
    详细可以在mo-net.py中搜索“解包”。"""
    mask = dataset.image_info[image_id]['mask']
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    # 上句，缩放图像。不过估计没啥影响，因为image原来就是512,512,3的，完了之后还是这个size。
    masks = resize_mask(masks, scale, padding)
    # 上句，缩放图像，输入的scale和padding是从上上句得到的。不过，同样似乎没啥影响。

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
            mask = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding masks got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(masks)
    # 上句，从掩膜计算外接矩形。shape=(3,4)，3表示有3个通道，4表示每个通道对应4个角点坐标。

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # 上句，active_class_ids此时是7（dataset.num_classes=7）个0。
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]  # 原来的
    # 原来的程序，上句source_class_ids此时是0~6这7个数（dataset是个HaNDataset类型变量，source_class_ids是其数据成员）。
    #     然后[]里面的dataset.image_info[image_id]["source"]是说，
    #         第image_id张图中的image_info数据成员中的下标为"source"的东西，应该要么是''要么是'organs'，（后来CT数据集，应该是要么是''、要么是'CT'、要么是'MRI'）
    #     所以上句等号右边就成了dataset.source_class_ids['']或者dataset.source_class_ids['organs']，
    #         所以就要么是[0]要么是[0,1,2,3,4,5,6]。
    # utils.py里的prepare函数里说了这个。
    source_class_ids = dataset.source_class_ids['organs']  # 改的
    # 检测数据集（dataset.image_info[image_id]["source"]不存在），索性直接弄成dataset.source_class_ids['organs']了，
    #     就是说这个数据集里可能有这10中东西，即0是背景类、加上1~9这9个椎骨类别。这应该是整个数据集的，而不是某一张图有没有这个类别的。
    active_class_ids[source_class_ids] = 1
    # 上句，active_class_ids是7个1。这玩意似乎表示，当前的数据集（应该不是当前这张图）中，这7种分类都有可能存在。
    #     后面输出的那个class_ids，才是这张图里有哪几个非背景类类别的。。

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        masks = minimize_mask(bbox, masks, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    # 上句，拼成所谓的meta，作为class MaskRCNN的输入

    return image, image_meta, class_ids, bbox, masks, mask



def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    给定锚点和金标准外接矩形，计算重合度，并且识别正例锚和外接矩形修正，来细化锚以匹配金标准外接矩形。

    anchors: [num_anchors, (y1, x1, y2, x2)]  输入anchor：锚，shape应该是(锚数, 4)，我们这儿是(65472, 4)。
    gt_class_ids: [num_gt_boxes] Integer class IDs.  输入gt_class_ids：shape应该是金标准外接矩形数目，某个样例中是(28,)。
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]  输入gt_boxes：shape应该是（金标准外接矩形数目，4），某个样例中是(28,4)。
    以上是输入参数，都是前面utils.generate_pyramid_anchors和load_image_gt函数的输出。

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
        输出rpn_match：锚和金标准外接矩形之间的匹配：1是正例，-1是负例，0是中性。
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        输出rpn_bbox：锚外接矩形的修正值。

    第二遍又看了一遍：
    知道它是这样用的：从“金标准类别gt_class_ids”和“外接矩形gt_boxes”，到“RPN类别金标准rpn_match”和“rpn外接矩形修正rpn_bbox”。
    第二遍想弄明白什么问题？
    1、输出rpn_bbox为什么是修正，修正干什么用的？
        【就是把某个正例锚，对应到离它最近的金标准外接矩形上，然后就看锚的那个矩形和外接矩形之间的位置（和长宽）差别，予以修正。
        这一步相当于是把图中的每个金标准，对应到了离他最近的正例锚上。】
    2、为什么要把金标准类别和外接矩形弄成rpn_match和rpn_bbox？
        【就是要知道，哪些锚是正例，然后他们和金标准有什么对应关系。因为正式训练的时候，也不是直接用金标准类别和掩膜算的损失，
        比如说预测出来一个提出（弄出来它的类别和外接矩形了），但是，原图中有好几个金标准呢，电脑怎么知道这个提出应该对应哪个金标准啊！
        所以要用那个锚点，把预测出来的那个提出对应到最近的锚点（应该是正例锚点吧）上，
        然后再找离那个锚点最近的金标准（就是那些target_class_ids、target_bbox什么的），这样就可以用那个金标准去修正那个类别和外接矩形了。
        所以说那些锚就是起一个定位作用的，找到和某个预测位置上最接近的金标准。】
    3、输出的rpn_match为何是±1和0，而不是类别索引号？
        【可以解释为这儿只需要知道它是正例还是负例，然后下一步选出正例去干后面的工作。但为什么要这样做呢？】
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)  # 现在是65472个0组成的向量
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))  # 现在是shape=(256, 4)的0向量。

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    # 现在这个拥挤矩形应该更明白了。这个gt_class_ids是输入进来的金标准分类号，如果按照那个样例说的，是[1,3,4]的话，那么crowd_ix就没有。
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
        # 最终得到这个no_crowd_bool，是个shape=(65472,)的矩阵，每个值都是0或者1，如果是1的话就说明相应的锚不是拥挤矩形。
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)  # 此种情况，65472个都是1，所有的锚都不是拥挤矩形。

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)
    # shape=(num_anchors,num_gt_boxes)的矩阵，每个锚和金标准外接矩形重合度。用b=np.max(overlaps)看了看，b得0.579，即最大重合度差不多是这个。

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # 上句，结果是个(65472,)的向量，其中大部分元素是0，但是也有几个不是的。
    #     就是说，这65472个数就是这么多个锚，然后每个数的值，就表示相应的锚和第几个金标准重合度最大啊。
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    # 上句，anchor_iou_max的shape=(65472,)，数据都是0~1之间的，因为是重合度啊。
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 上句，rpn_match仍然是65472个数的向量，只不过“与金标准外接矩形重合度小于<0.3且非拥挤矩形”这样的位置中的数，都被写成了-1，
    #     这些位置对应的锚就叫做负样本。

    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)  # “和金标准外接矩形重合度最大的锚”的位置，如62929。
    rpn_match[gt_iou_argmax] = 1  # 该位置的样本设为正样本
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1
    # 上句，如果有的锚“和金标准外接矩形重合度”>0.7，那么这些锚就设为正样本。
    #     2和3两个都执行，是为了保证所有锚和金标准重合度都小于0.7的时候，也能至少有一个正样本。
    # rpn_match_is_1 = np.where(rpn_match==1)
    # if rpn_match_is_1[0].shape[0] > 3 :
    #     print('有3个以上的锚点被认为是RPN正例。')  # 看看这种情况，RPN的金标准和预测值是什么样子？写在下面。。。
    """以上，总体的目的是为了给每个金标准外接矩形，找到一个锚（Set an anchor for each GT box）。
    也就是说，要通过overlaps（包含金标准gt_boxes和锚anchor的位置信息）找到所有的正样本，
        并且，把rpn_match中、正样本索引号对应的位置标记为1。
    上面，“有3个以上的锚点被认为是RPN正例”的情况，是这样的————
        金标准：那个rpn_match就是3个以上等于1的（3个以上的RPN正例啊）了，没啥好说的。
            那个金标准RPN外接矩形修正值就是那几个RPN正例的外接矩形修正值，其他的都是0了。
            （如果有3个以上的RPN正例，那么这个金标准RPN外接矩形也会有3个以上不为0的情况了）
            另外这个RPN里的金标准外接矩形修正值，和mrcnn里的那个外接矩形修正值是不一样的。即使是训练了很长时间也是不一样的。
        预测值：RPN_class就是那65472个锚中的每一个，为负例和正例的概率。
            外接矩形就是(2, 65472, 4)，是这65472个都有的，然后我发现预测为正例的那几个，确实是会逐渐接近于金标准的外接矩形修正值。
    """

    # Subsample to balance positive and negative anchors  下采样，平衡正例和负例
    # Don't let positives be more than half the anchors  不要让正例多于锚数目的一半
    ids = np.where(rpn_match == 1)[0]  # 正例的索引，如刚才的那个62929。
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    # 多了的正样本的个数。就是正样本数len(ids)-总样本数的一半。现在由于正样本就1个，所以这个extra是个负数。
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0# 这是随机地把多余的正样本弄成中性样本。当然上面的情况，这个if里的东西都没执行。
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]  # 负例的索引，一般都有一大堆的负例。比如现在就有65460个（说明这个样本负例严重超了啊）
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    # 上句，负例本来应该有config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1)（即总样本数-正例数）个，实际上有len(ids)个，就多了一些。
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0  # 把多余的负例弄成中性的了

    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    # 对于每个正例样本，计算位移和放缩量，变换他们，让他们和对应的金标准外接矩形匹配。
    ids = np.where(rpn_match == 1)[0]  # 正例样本索引号。
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):  # 这个for循环，i是ids中的一个数（即某个正例的索引号），anchors[ids]就是该正例（即锚）的角点坐标。
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]  #
        """上句，找到离当前正例（第i个正例）最近的金标准外接矩形。
        注意i只是在ids里循环，也就是说只是正例的序号，如62929、62930等等，而并不是从0~65471的那些。
        然后，anchor_iou_argmax[i]就表示，离第i个样本（因为i属于ids，所以这个样本肯定是正例）最近的金标准外接矩形。
        """

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w  # 以上，计算出来金标准和锚（即那个正例）的长宽、中心点坐标。

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [(gt_center_y - a_center_y) / a_h, (gt_center_x - a_center_x) / a_w, np.log(gt_h / a_h), np.log(gt_w / a_w),]
        # 上句，计算出来外接矩形修正的那些数。
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV  # 正则化
        ix += 1
        """
        以上，把锚匹配到金标准外接矩形中，即先从65472个锚中把正例选出来，然后找到其中的正例对应的金标准外接矩形，最终计算出需要修正的值（rpn_bbox）
        忽然想起来一个事儿，class MaskRCNN里的【从图片到提出的第五步】，
            在detection_targets_graph函数里也去除了拥挤矩形，然后弄了正例负例什么的。这里有什么不一样的吗？
            →→→那个是以下几步：
                ①先从所有金标准提出中，拿出非拥挤矩形的金标准类别ID、外接矩形、掩膜；
                ②计算所有提出和金标准外接矩形重合度，还有他们和拥挤矩形的重合度；
                ③根据各个提出和金标准外接矩形的重合度，选择正例和负例；
                ④下采样，即根据指定的正例和负例个数，去在正例和负例中选择这么多个；
                ⑤把刚刚弄出来的每一个正例，对应到了某个金标准提出上；
                ⑥计算外接矩形修正值、弄对应掩膜什么的。
            目的首先就是不一样的，这个是要弄金标准正例和负例，那是要在已知金标准锚是正例还是负例还是拥挤矩形的情况下，
                来弄RPN提出的正例和负例，然后把正例提出和相应的金标准提出对应起来。
            然后那个是先把锚细化为提出，并且有下采样、弄掩膜的步骤，这个没有。
            还有就是，这儿是弄出是正例还是负例还是中性，而class MaskRCNN里的【从图片到提出的第三步】做了类似事情，得到那个rpn_class_logits，
                与这儿输出的金标准input_rpn_match对比得到损失啊。
            类似的地方是，正例和负例都是根据重合度选择的、都计算了外接矩形修正值（以备后续程序使用）。
        """
    return rpn_match, rpn_bbox



def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    把那个0~255之间的RGB值，减掉该通道的平均值，并且弄成float32类型。
    然后那个平均值，是在config.py中直接给出来的。但是难道不是当前图去作平均吗？
    """
    return images.astype(np.float32) - config.MEAN_PIXEL



def batch_processing(process_func, input_batches, **kwargs):
    """现在试试能否输入也是多个张量。上一个函数里的input_batch变成了input_batches，就是说输入的东西是多个张量了。
    当然，要求每个张量的第0维（即有几个矩阵）都相等。
    """
    if not isinstance(input_batches, list):  # 先把输入的np.array变成list。
        input_batches = [input_batches]
    processed_all = []  # 这样，就不需要写一大堆的(参数名_all)这样的变量，然后一个一个append在concatenate了。
    for i in range(input_batches[0].shape[0]):
        slices = [x[i] for x in input_batches]  # 批次中的、一张图中的所有输入张量。
        processed = process_func(*slices, **kwargs)  # 现在slices未必只有1个张量，所以是*slices。然后后面的**kwargs不变。
        if not isinstance(processed, (tuple, list)):  # 如果output_slice不是tuple或者list类别，就变成list。
            processed = [processed]
        processed_all.append(processed)
    processed_all_zipped = list(zip(*processed_all))
    # 上句，processed_all_zipped是个list，里面的每一个元素都是tuple（这个时候好像不太方便给他变成np.array，毕竟这个函数
    #     不知道process_func的输入输出，也就不知道他有几个元素啊）。
    result = [np.stack(o, axis=0) for o in zip(processed_all_zipped)]
    # 上句，弄成list，然后主函数里可以直接用，见调用的时候。但是调用的时候就会发现，比起单个输出的，就都多了一维。不过，可以在这儿就把它删掉。
    result = [np.squeeze(r) for r in result]  # 删掉多余的维度（是前面的zip等操作搞出来的）
    if len(result) == 1:
        result = result[0]
    return result



def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    把输入分成切片，然后把每个切片赋给给出的计算图的，然后结合结果。
    这允许你在一个批次的输入中跑计算图，即使这个计算图只支持一个样例。

    inputs: list of tensors. All must have the same first dimension length
        输入的inputs，一系列张量，所有的张量的第1维必须长度相等
    graph_fn: A function that returns a TF tensor that's part of a graph.
        输入的graph_fn，一个函数，返回一个TF张量
    batch_size: number of slices to divide the data into.
        批次大小：把数据集分成的切片数。
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):  # isinstance用来判断某个变量是不是某种类型。这儿的意思是，inputs如果是list类型，那么isinstance(inputs, list)就是1。
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]  # 从输入中选取第i张图的各个变量。输入[scores, ix]的第一维本来都是?（一个批次中不知道几张图），
        #                                            但现在由于batch_size是用config.IMAGES_PER_GPU给出来的，故i就是从0~3，这也就是后面的那个4的来源。
        output_slice = graph_fn(*inputs_slice)  # 这句应该是执行那个graph_fn函数（即detection_targets_graph），然后*inputs_slice应该就是inputs中的那4个选中了的元素。
        if not isinstance(output_slice, (tuple, list)):  # 如果output_slice不是tuple或者list类别，就变成list。
            output_slice = [output_slice]
        outputs.append(output_slice)  # 每循环一次，这个outputs（是个list）就多一个shape为(6000,)的张量。
        #                                   i是从0~3的话，此时outputs就是个长度为4的list。
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))  # 这里outputs仍然是list，但是长度变成1了。

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]  # 这是把list变成张量，所以4个(6000,)的张量就变成了shape=(4,6000)的张量。
    if len(result) == 1:
        result = result[0]

    return result



def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    把给出的外接矩形角点，应用到给出的外接矩形上去。
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # 以上四句是得到外接矩形的长宽和中心点。

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # 以上四句是修正。中心点是加了一个距离（deltas中的一个元素）；长宽是乘了一个常数。

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    # 以上四句是把修正后的长宽和中心点变回角点坐标。

    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result



def clip_boxes_graph(boxes, window):
    """
    这个就是个很简单的裁剪，如果x或y坐标小于0或者大于最大值，就变成0或者最大值。
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped



def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    分析一个向量，这个向量包含图像属性...
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    输入meta，[批大小，meta长度]

    其实就是把那个meta解码称为图像的序号、形状、窗口、活跃的ID。具体见data_generator函数里关于image_meta的说明。
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]



def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    通常，外接矩形是用shape为[N,4]的矩阵表示的，并且被补了0（这是因为很多图中没有N个提出啊）。
        本函数就是除掉那些补了的零。
    输入的boxes是[N,4]的矩阵吗，是

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    # 上句是把那个[N,4]的矩阵，每一行的4个数加起来了，然后转化为bool形式，
    #     就是说如果这4个数不都是0，那加起来就不是0，bool了之后就变成1了。
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    # 上句是从boxes中选出non_zeors为Ture的那些元素。
    # 例如boxes是[[a1,b1,c1,d1],[a2,b2,c2,d2],[0,0,0,0]]，那么non_zeros就应该是[1,1,0]，
    #     然后上句就输出[[a1,b1,c1,d1],[a2,b2,c2,d2]]。
    return boxes, non_zeros
def trim_zeros_graph_labels(labels, name=None):
    """
    这个是给1维的标签去掉补零用的。
    labels: [N] 标签.
    non_zeros: [N] 哪些位置是非零的（用True表示）。
    """
    non_zeros = tf.cast(labels, tf.bool)
    labels = tf.boolean_mask(labels, non_zeros, name=name)
    return labels, non_zeros


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    计算两个矩形的IOU。应该是任意的矩形都可以。细节先不看了。
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)  # 两个矩形的交集
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection  # 两个矩形的并集
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])  # 为啥要reshape一下？？？
    return overlaps



def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    得到从“外接矩形”到“金标准外接矩形”需要细化的值。
    和model.py里的那个apply_box_deltas_graph有点像，但是那个是用这个delta去修正，而现在是根据金标准和提出的差异，去得到这个delta。
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result



def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row in x depending on the values in counts.
    根据counts里的值（每一张图中，正例锚的个数），从x（输入的金标准RPN外接矩形，是(batch_size, 256, 4)的张量）的每张图中
        分别选出相应的counts[i]个，得到一个list，然后把list变成张量。
        说得高大上，其实就是把输入x的每个分量的前counts[i]个（因为这前面的几个就对应正例的外接矩形）取出来拼在一起。

    调用的时候输入的x是个(?,?,4)的张量，counts是个(?,)的张量。
    """
    outputs = []
    for i in range(num_rows):  # 这个i是对若干张图做循环的
        outputs.append(x[i, :counts[i]])  # x[i, :counts[i]]是第i列的，从0到counts[i]的所有元素。
    return tf.concat(outputs, axis=0)



def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss



def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
        输入的掩膜是小的，比如，28*28的掩膜。确实是mask的shape是(28, 28)
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
        输入的bbox：要把掩膜适应到这个外接矩形里面。
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    # 上句，mask变成了shape=(y2 - y1, x2 - x1)的矩阵，其中的每一个数仍然是这一点属于这一类的概率。
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
    # 上句，shape还是(y2 - y1, x2 - x1)，里面的数变成0或者1了，就是说，如果概率大于threshold就认为是这一类，否则就认为不是
    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)  # 这玩意的shape是(512, 512)，相当于是原图的大小
    full_mask[y1:y2, x1:x2] = mask  # 这个是把那个mask按照y1, x1, y2, x2的位置，放到原图中去。有一次等号左右两边的shape不一样，就报错了。
    return full_mask
def unmold_mask_tf(mask, bbox, image_shape):
    """上面函数的tf版，最好弄到某个函数里去试试啊。。。主要是scipy.misc.imresize和tf.image.resize_images不知道差得多不多啊。。"""
    threshold = 0.5
    y1 = bbox[0]
    x1 = bbox[1]
    y2 = bbox[2]
    x2 = bbox[3]
    mask = tf.image.resize_images(mask, (y2 - y1, x2 - x1), method=0)  # method=0似乎就表示双线性插值
    mask = tf.cast(mask, tf.float32)
    mask = tf.where(mask >= threshold, 1, 0)  # 好像不对啊？？？？
    mask = tf.cast(mask, tf.uint8)
    full_mask = tf.zeros(image_shape[:2], dtype=tf.uint8)  # 可能要image_shape.as_list()[1:3]
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def kick_off_repeat_targets(target_class_ids, mrcnn_class_logits, mrcnn_class, rois_or_rpn_rois, target_bbox, target_mask, mrcnn_bbox, mrcnn_mask):
    """有的时候，DetectionTargetLayer的执行结果会出来不止3个target_class_ids，导致后面训练会用错误的target_class_ids作为金标准。
    这儿想根据mrcnn_class_logits，删掉一些不太可能的target_class_ids。然后，剩下别的东西都要给更新一下的。
    如果批大小是2，每张图里有100个提出，那么——
    target_class_ids---(2, 100)
    mrcnn_class_logits---(2, 100, 4)
    rois_or_rpn_rois---(2, 100, 4)
    target_bbox---(2, 100, 4)
    target_mask---(2, 100, 28, 28)
    mrcnn_bbox---(2, 100, 4, 4)
    mrcnn_mask---(2, 100, 28, 28, 7)
    这个函数里，这些东西都是np的，要用那个

    但是忽然又觉得，是不是还是用overlaps比用这个mrcnn_class_logits做要好一些？？因为我隐隐感觉，用一个预测的值去修改金标准的值，似乎有点不对劲儿。
    两个都试试吧。。。
    """
    target_class_ids_updated = np.zeros_like(target_class_ids, dtype=np.int32)  # 一开始都弄成0
    mrcnn_class_logits_updated = copy.copy(mrcnn_class_logits)  # 一开始都弄的和mrcnn_class_logits一样。不太确定。。
    mrcnn_class_updated = copy.copy(mrcnn_class)  # 一开始都弄的和mrcnn_class_logits一样。不太确定。。
    rois_or_rpn_rois_updated = copy.copy(rois_or_rpn_rois)  # 一开始都弄的和rois_or_rpn_rois一样。不太确定。。
    target_bbox_updated = np.zeros_like(target_bbox, dtype=np.float32)  # 一开始都弄成0。不太确定。。
    # 后来感觉好像target有关的，一开始还是弄成0比较好。相当于就是把那些多弄出来的targets(金标准)都给删了。
    target_mask_updated = np.zeros_like(target_mask, dtype=np.float32)  # 一开始都弄成0。不太确定。。
    mrcnn_bbox_updated = copy.copy(mrcnn_bbox)  # 一开始都弄的和mrcnn_bbox一样。不太确定。。
    mrcnn_mask_updated = copy.copy(mrcnn_mask)  # 一开始都弄的和mrcnn_mask一样。不太确定。。
    try:  # 大多数都是能实现的
        for i in range(target_class_ids.shape[0]):
            target_class_id = target_class_ids[i, :]
            mrcnn_class_logit = mrcnn_class_logits[i, :, :]
            mrcnn_class_this = mrcnn_class[i, :, :]
            roi_or_rpn_roi = rois_or_rpn_rois[i, :, :]
            target_bbox_this = target_bbox[i, :, :]
            target_mask_this = target_mask[i, :, :, :]
            mrcnn_bbox_this = mrcnn_bbox[i, :, :, :]
            mrcnn_mask_this = mrcnn_mask[i, :, :, :, :]
            non_zero_pos = np.where(target_class_id > 0)[0]
            target_class_id_not_zero = target_class_id[non_zero_pos]  # 就是那几个非零元素，如，array([2, 3, 1, 3])。
            if target_class_id_not_zero.shape[0] > 3:  # 一开始可能target_class_ids里什么都没有，所以只有发现了多于3个的，才做后面的工作。
                num_classes = np.max(target_class_id_not_zero)  # 一共有几种？这儿就是3了。原来是放在if前面的，现在放这儿吧。。
                mrcnn_class_logit_non_zero = mrcnn_class_logit[non_zero_pos, :]
                mrcnn_class_this_non_zero = mrcnn_class_this[non_zero_pos, :]
                roi_or_rpn_roi_non_zero = roi_or_rpn_roi[non_zero_pos, :]
                target_bbox_this_non_zero = target_bbox_this[non_zero_pos, :]
                target_mask_this_non_zero = target_mask_this[non_zero_pos, :, :]
                mrcnn_bbox_this_non_zero = mrcnn_bbox_this[non_zero_pos, :, :]
                mrcnn_mask_this_non_zero = mrcnn_mask_this[non_zero_pos, :, :, :]
                mrcnn_logit_sorted_pos_ori_max = []
                mrcnn_logit_sorted_pos_ori_oth = []
                for j in range(1, num_classes + 1):  # 第0类是背景，不管他，从第1类开始。这儿j是类别序号。
                    this_id_pos = np.where(target_class_id_not_zero == j)[0]  # 这一类别的所有提出，在原来的非零提出矩阵中的位置
                    this_id_mrcnn_logit = mrcnn_class_logit_non_zero[this_id_pos, j]  # 这一类别的所有mrcnn_class_logit。
                    # 注意是[this_id_pos, j]而不是[this_id_pos, :]，因为mrcnn_class_logit_non_zero是所有类别的logits，而我现在只按照这一类别排序。
                    # this_id_mrcnn_logit_sorted = np.sort(this_id_mrcnn_logit)  # 这一类的logit从小到大排列。不知道为什么变成屎黄色的，但是看起来没什么问题。。。
                    # this_id_mrcnn_logit_sorted = this_id_mrcnn_logit_sorted[::-1]  # 倒序过来，从大到小。
                    this_id_mrcnn_logit_sorted_pos = np.argsort(this_id_mrcnn_logit)  # 这一类的logit从小到大排列后的顺序
                    this_id_mrcnn_logit_sorted_pos = this_id_mrcnn_logit_sorted_pos[::-1]  # 倒序过来，从大到小。
                    this_id_mrcnn_logit_sorted_pos_ori = this_id_pos[this_id_mrcnn_logit_sorted_pos]
                    # 这一类的logit从大到小排列后的，在原来的非零提出矩阵中的位置
                    this_id_mrcnn_logit_sorted_pos_ori_max = this_id_mrcnn_logit_sorted_pos_ori[0]
                    # 这一类的logit最大的提出，在原来非零提出矩阵中的位置
                    # 上句输出会是一个数
                    this_id_mrcnn_logit_sorted_pos_ori_oth = this_id_mrcnn_logit_sorted_pos_ori[1:]
                    # 这一类的logit其他提出，在原来非零提出矩阵中的位置
                    # 上句输出要么是[]（空集），要么是array([6], dtype=int64)这样的集合。
                    mrcnn_logit_sorted_pos_ori_max.append(this_id_mrcnn_logit_sorted_pos_ori_max)
                    mrcnn_logit_sorted_pos_ori_oth.append(this_id_mrcnn_logit_sorted_pos_ori_oth)
                mrcnn_logit_sorted_pos_ori_max_arr = np.stack(mrcnn_logit_sorted_pos_ori_max)  # logits最大的弄成np.array
                mrcnn_logit_sorted_pos_ori_oth_arr = np.concatenate(mrcnn_logit_sorted_pos_ori_oth)  # 其他的弄成np.array
                mrcnn_logit_sorted_pos_final = np.concatenate(
                    (mrcnn_logit_sorted_pos_ori_max_arr, mrcnn_logit_sorted_pos_ori_oth_arr))
                # 上句就是最终想要的顺序。。
                target_class_id_not_zero_sorted = target_class_id_not_zero[mrcnn_logit_sorted_pos_final]
                mrcnn_class_logit_non_zero_sorted = mrcnn_class_logit_non_zero[mrcnn_logit_sorted_pos_final, :]
                mrcnn_class_this_non_zero_sorted = mrcnn_class_this_non_zero[mrcnn_logit_sorted_pos_final, :]
                roi_or_rpn_roi_non_zero_sorted = roi_or_rpn_roi_non_zero[mrcnn_logit_sorted_pos_final, :]
                target_bbox_this_non_zero_sorted = target_bbox_this_non_zero[mrcnn_logit_sorted_pos_final, :]
                target_mask_this_non_zero_sorted = target_mask_this_non_zero[mrcnn_logit_sorted_pos_final, :, :]
                mrcnn_bbox_this_non_zero_sorted = mrcnn_bbox_this_non_zero[mrcnn_logit_sorted_pos_final, :, :]
                mrcnn_mask_this_non_zero_sorted = mrcnn_mask_this_non_zero[mrcnn_logit_sorted_pos_final, :, :, :]
                target_class_id_not_zero_sorted[num_classes:] = 0
                target_bbox_this_non_zero_sorted[num_classes:, :] = 0
                target_mask_this_non_zero_sorted[num_classes:, :, :] = 0  # 把logits非最大的清零，保证只有1个金标准。
                # 以上是按照最终想要的顺序，排列前面那些非零的序号、logits、类别、外接矩形、掩膜，并且把logits非最大的target给清零，保证后面用的金标准中每一种都有1个。
                non_zero_len = target_class_id_not_zero_sorted.shape[0]
                assert (non_zero_len == mrcnn_class_logit_non_zero_sorted.shape[0])
                assert (non_zero_len == mrcnn_class_this_non_zero_sorted.shape[0])
                assert (non_zero_len == roi_or_rpn_roi_non_zero_sorted.shape[0])
                assert (non_zero_len == target_bbox_this_non_zero_sorted.shape[0])
                assert (non_zero_len == target_mask_this_non_zero_sorted.shape[0])
                assert (non_zero_len == mrcnn_bbox_this_non_zero_sorted.shape[0])
                assert (non_zero_len == mrcnn_mask_this_non_zero_sorted.shape[0])
                target_class_ids_updated[i, :non_zero_len] = target_class_id_not_zero_sorted
                mrcnn_class_logits_updated[i, :non_zero_len, :] = mrcnn_class_logit_non_zero_sorted
                mrcnn_class_updated[i, :non_zero_len, :] = mrcnn_class_this_non_zero_sorted
                rois_or_rpn_rois_updated[i, :non_zero_len, :] = roi_or_rpn_roi_non_zero_sorted
                target_bbox_updated[i, :non_zero_len, :] = target_bbox_this_non_zero_sorted
                target_mask_updated[i, :non_zero_len, :, :] = target_mask_this_non_zero_sorted
                mrcnn_bbox_updated[i, :non_zero_len, :, :] = mrcnn_bbox_this_non_zero_sorted
                mrcnn_mask_updated[i, :non_zero_len, :, :, :] = mrcnn_mask_this_non_zero_sorted
            else:
                target_class_ids_updated[i, :] = target_class_ids[i, :]
                mrcnn_class_logits_updated[i, :, :] = mrcnn_class_logits[i, :, :]
                mrcnn_class_updated[i, :, :] = mrcnn_class[i, :, :]
                rois_or_rpn_rois_updated[i, :, :] = rois_or_rpn_rois[i, :, :]
                target_bbox_updated[i, :, :] = target_bbox[i, :, :]
                target_mask_updated[i, :, :, :] = target_mask[i, :, :, :]
                mrcnn_bbox_updated[i, :, :, :] = mrcnn_bbox[i, :, :, :]
                mrcnn_mask_updated[i, :, :, :, :] = mrcnn_mask[i, :, :, :, :]
    except:
        target_class_ids_updated = target_class_ids
        mrcnn_class_logits_updated = mrcnn_class_logits
        mrcnn_class_updated = mrcnn_class
        rois_or_rpn_rois_updated = rois_or_rpn_rois
        target_bbox_updated = target_bbox
        target_mask_updated = target_mask
        mrcnn_bbox_updated = mrcnn_bbox
        mrcnn_mask_updated = mrcnn_mask
    return target_class_ids_updated, mrcnn_class_logits_updated, mrcnn_class_updated, rois_or_rpn_rois_updated, \
           target_bbox_updated, target_mask_updated, mrcnn_bbox_updated, mrcnn_mask_updated


##################################################
#            以下是测试的时候用到的函数             #
##################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)



def mold_inputs(config, images):  # 在后面detect函数里调用过。
    """Takes a list of images and modifies them to the format expected as an input to the neural network.
    输入一个list的图像，把它们变成某种形式，这种形式作为神经网络的输入。看来这个是要修改的函数啊。。。
    images: List of image matricies [height,width,depth]. Images can have different sizes.
        输入的images：一个图像矩阵[height,width,depth]组成的list，图像可能有不同的大小。

    Returns 3 Numpy matricies: 返回3个numpy矩阵：
    molded_images: [N, h, w, 3]. Images resized and normalized.
        返回的molded_images：缩放并且正则化了的图像，是一个张量，N张h*w的图片，3是表示是RGB图。
    image_metas: [N, length of meta data]. Details about each image.
        返回的image_metas：一个张量，N是图片张数，length of meta data是图像meta的长度。
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the original image (padding excluded).
        返回的windows：有原始图像的图像窗口（就是把图像外面的补零去掉了之后的东西），四个坐标值，之内的图是有用的，外面的是0。
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image to fit the model expected size
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding = resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)  # self.config.IMAGE_PADDING现在是True，就是要补零的。
        """上句，放缩并补零图像，返回补零后的图、原图窗口、放缩尺度（因为是放缩到所需大小的，所以这个尺度是预先不知道的）、补零的地方。
        似乎，弄完了之后，图的大小都变成了self.config.IMAGE_MAX_DIM*self.config.IMAGE_MAX_DIM。"""
        molded_image = mold_image(molded_image, config)
        """上句，就是把图像减掉三个通道的平均值。"""
        """
        这句话不确定要不要！！！！似乎是更应该删掉的。。。
        在MaskRCNN_0_get_inputs.py里的get_batch_inputs_for_MaskRCNN中，
        这句话image = MRCNN_utils.mold_image(image.astype(np.float32), config)是注释掉了的
        """
        # Build image_meta
        image_meta = compose_image_meta(0, image.shape, window, np.zeros([config.NUM_CLASSES], dtype=np.int32))
        """上句，把图像的形状、窗口、类别编码。详见那个函数。
        不过不明白为啥第一个（图像序号）是0、为啥最后那个也是7个0（似乎应该是7个1啊）"""
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
        """以上，把这张图的信息添加到总信息中。"""
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    # 以上是把list变成array，然后返回。
    return molded_images, image_metas, windows



def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)
def norm_boxes_tf(boxes, shape):
    """就是上面的归一化函数，tf版本
    输入的boxes可以是np（比如boxes就是那个window，np.array([  0,   0, 512, 512])，shape是），
        似乎boxes也可以是tf，（下面那个denorm_boxes_tf就是tf）。shape应该不能是np，否则第一步h, w = shape就会报错了。
    """
    h, w = shape
    scale = tf.cast(tf.constant([h - 1, w - 1, h - 1, w - 1]), tf.float32)  # np.array改成tf.constant，注意有的时候会有数据类型问题，如果报错就加一个cast。
    """【重要！！】就看出来一个结论，似乎可以把表示一些数的np.array用tf.constant代替，当然要注意数据类型。
    然后不代替似乎也可以，因为np和tf是可以相乘或者相加的。"""
    shift = tf.cast(tf.constant([0, 0, 1, 1]), tf.float32)
    normed_boxes = ((boxes - shift)/scale)
    normed_boxes = tf.cast(normed_boxes, tf.float32)
    return normed_boxes



def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
def denorm_boxes_tf(boxes, shape):
    """反归一化，tf版本
    这儿输入的boxes是tf，shape是np。
    """
    h, w = shape
    scale = tf.cast(tf.constant([h - 1, w - 1, h - 1, w - 1]), tf.float32)  # 注意数据类型！
    shift = tf.cast(tf.constant([0, 0, 1, 1]), tf.float32)
    denormed_boxes =tf.round((boxes*scale) + shift)
    """【重要！！】这儿就说明，np和tf是可以相乘或者相加的，只要满足点乘/加的条件，即矩阵维度一样。然后弄完了就得到的是tf。"""
    denormed_boxes = tf.cast(denormed_boxes, tf.int32)
    return denormed_boxes


def get_patient_ids(dataset):
    num_examples = len(dataset.image_info)
    dataset_patiend_ids = []
    for i in range(num_examples):
        patiend_id_this = dataset.image_info[i]['patient']
        dataset_patiend_ids.append(patiend_id_this)
    dataset_patiend_ids = np.array(dataset_patiend_ids)
    return dataset_patiend_ids


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_coord_error(gt, pred):
    """计算金标准和预测外接矩形的误差。为了和中心点预测的对比用的。"""
    num_gt = gt.shape[0]
    num_pred = pred.shape[0]
    if num_gt > num_pred:  # 其实是很sb的事情，如果个数都tmd不一样，怎么算这个误差，烦死了。。
        diff = num_gt - num_pred
        pad = np.zeros([diff, 4], dtype=np.int32)
        pred = np.concatenate([pred, pad], axis=0)
    elif num_pred > num_gt:
        diff = num_pred - num_gt
        pad = np.zeros([diff, 4], dtype=np.int32)
        gt = np.concatenate([gt, pad], axis=0)
    indices_gt = np.argsort(gt[:, 0])  # 预测的外接矩形按照第0列（y1值）排序
    gt_sorted = gt[indices_gt][::-1]
    indices_pred = np.argsort(pred[:, 0])  # 预测的外接矩形按照第0列（y1值）排序
    pred_sorted = pred[indices_pred][::-1]
    coord_error_this = np.mean(abs(gt_sorted - pred_sorted))
    return coord_error_this


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids



def compute_ap(num_classes, gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    计算在给定IOU阈值时的AP。默认IOU阈值是0.5（似乎就是所谓的AP50吧）

    Returns:
    mAP: Mean Average Precision
        按照coco的网站（http://cocodataset.org/#detection-eval），这个mAP似乎是指各个种类之间的平均精确度。
    precisions: List of precisions at different class score thresholds.
        不同类别分数阈值的精确度列表
    recalls: List of recall values at different class score thresholds.
        不同类别分数阈值的召回率列表
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        重合度矩阵：应该是有pred_boxes行、gt_boxes列，然后第i行第j列的数就表示第i个预测外接矩形和第j个金标准外接矩形的重合度。
        怎么觉得有点别扭啊？我一个预测的提出，和所有金标准提出都有个IOU，然后我就取最大的IOU当做评估准则么？这下也不管类别对不对了么？？？
    """
    # Trim zero padding and sort predictions by score from high to low  裁剪掉补零，并且从高到低排序预测分数。
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]  # :pred_boxes.shape[0]是从0到pred_boxes的第一个维度（即有多少个预测外接矩形）
    """以上三句，先除掉金标准外接矩形和预测外接矩形后面的0（找到有效的金标准/预测外接矩形），然后找到对应的预测分数。"""
    indices = np.argsort(pred_scores)[::-1]  # indices是把所有提出按照预测分数从高到低排列，排列好后的索引号。
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    """以上四句，先按照分数高低排序索引号，然后把预测的外接矩形、类别、分数都按此顺序排序。"""

    # Compute IoU overlaps [pred_boxes, gt_boxes]  计算IOU值，每一行就是一个预测的外接矩形和所有金标准外接矩形的IOU。比如说，如果有19行，每行18个数（在View as array中看起来，竖着的序号是0~18这19个，横着的是0~17这18个），那就说明预测出来19个外接矩形，然后第i行就是第i个预测外接矩形和所有18个金标准外接矩形的IOU。大概扫了一眼，每一行的18个数的最大值，一般是在0.5~0.9之间，一般是0.7~0.8的。
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through ground truth boxes and find matching predictions  对金标准外接矩形循环，找到匹配的预测。
    match_count = 0  # 应该是类别判断对了的个数。现在初始化为0。
    pred_match = np.zeros([pred_boxes.shape[0]])  # 似乎pred_match和pred_boxes是等长的。即，现在是“预测外接矩形个数”个0。
    gt_match = np.zeros([gt_boxes.shape[0]])  # 现在是“金标准外接矩形个数”个0。
    for i in range(len(pred_boxes)):  # i是对每个预测外接矩形循环
        # Find best matching ground truth box  找到最佳匹配的金标准外接矩形。
        sorted_ixs = np.argsort(overlaps[i])[::-1]  # overlaps[i]是重合就在中的第i行，即第i个预测外接矩形和各个金标准的重合度。然后按照IOU大小去排序，IOU最大的序号放在最前面，例如，如果当前预测外接矩形和第7个金标准外接矩形重合度最大，那么sorted_ixs的第0个数就是7
        for j in sorted_ixs:  # sorted_ixs是排序好的金标准外接矩形索引号，所以这个j是对金标准外接矩形循环的，从重合度最大的金标准到重合度最小的金标准去循环，然后到某个阈值就停止循环（见下）。
            # If ground truth box is already matched, go to next one  如果金标准矩形已经匹配了，弄下一个。
            if gt_match[j] == 1:
                continue  # 如果当前金标准外接矩形已经被之前的某个外接矩形匹配上了，那么重新对j执行循环。
            # If we reach IoU smaller than the threshold, end the loop  如果碰上的iou小于阈值，停止循环。
            iou = overlaps[i, j]  # 当前金标准（第j个排序好的）和当前预测外接矩形（第i个）的IOU。
            if iou < iou_threshold:
                break
            # Do we have a match?  如果匹配的情况
            if pred_class_ids[i] == gt_class_ids[j]:  # 如果当前预测外接矩形和当前金标准外接矩形的类别一样（即所谓匹配），那么执行下面命令。如果没有匹配上，那么j就+1后继续（其实，一般j加了1之后，IOU往往下降很厉害，就小于那个阈值而跳出去了）。
                match_count += 1  # 匹配个数+1。
                gt_match[j] = 1  # 当前金标准外接矩形置1（注意前面有个“if gt_match[j] == 1”，就是说如果某个金标准外接矩形已经和之前的某个预测外接矩形匹配了，那么就不再弄了）。
                pred_match[i] = 1  # 当前预测外接矩形置1。
                break  # 退出j的循环，直接弄下一个预测外接矩形（下一个i）。

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)  # 分子np.cumsum是求累积和，如，输入是[1,1,0,0,1]，则输出是[1,2,2,2,3]。分母就是从1~预测外接矩形个数（如19）。所以除出来的东西，似乎和周老师书上说的向上走一步、向右走一步有点像，然后最后那个数应该是总的正确率。
    # 发现上面precisions向量中最后一个数，就是match_count/len(pred_class_ids)，相当于是类别判断的正确率。
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)  # 分子同上，分母就是一个数（金标准的长度）

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])  # 在precisions前后各添一个0。
    recalls = np.concatenate([[0], recalls, [1]])  # 在recalls前面添一个0，后面添一个1。

    # Ensure precision values decrease but don't increase. This way, the  确保准确度下降而不是上升，这样，在每个召回率阈值处的准确度就是其后的所有召回率阈值里的最大值。这玩意算的可能是这个网上说的那个“准确度-召回率曲线”：http://blog.sina.com.cn/s/blog_9db078090102whzw.html
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):  # i是从19到-1，往下降的。
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])  # 从后往前数，如果发现前面的数比后面的小，那就用后面的去替换掉。比如，如果是[...0.8125    , 0.82352941, 0.83333333, 0.78947368...]，那么会变成[...0.83333333, 0.83333333, 0.83333333, 0.78947368...]

    # Compute mean AP over recall range  计算召回率范围内的平均AP。
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1  # recalls[:-1]是recalls从第0个到倒数第二个；recalls[1:]是从第1个到最后一个。现在就是找上面二者的值不一样的地方（相当于是把recalls向量中重复的值都删掉了），记录下来索引号。
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    # 在机器视觉目标检测领域，AP和mAP分界线并不明显，都是多类检测的精度，关注点在检测框的精度上。在不同的recall水平上，计算了平均的准确率。
    # 这个网站：http://nooverfit.com/wp/david9的普及贴：机器视觉中的平均精度ap-平均精度均/
    # 或者这个：http://blog.sina.com.cn/s/blog_9db078090102whzw.html
    # 我们这儿，其实是首先做了个准确率和召回率的曲线，就是那个precisions和recalls矩阵，然后算了那个mAP值。
    # recalls[indices]就是把那个recalls里重复的都删了，然后第0个0也删掉，最后的1保留；recalls[indices - 1]也是把重复的都删掉，第0个0保留，最后的1删掉。
    # 然后(recalls[indices] - recalls[indices - 1])似乎是实现了加权平均的功能，然后那个乘号，相当于就是某个柱子的宽度*某个柱子的高度啊，然后求和，就是所有柱子的总面积啊。

    classification = match_count/len(pred_class_ids)
    match = np.where(pred_match == 1)  # 分类正确的外接矩形的索引（指的是在预测外接矩形中的索引号）
    overlap_match = overlaps[match,:]
    overlap_for_each_propasal = np.max(overlap_match,axis=2)
    overlap_mean = np.average(overlap_for_each_propasal)

    classes = (np.arange(num_classes - 1) + 1)
    APc = []
    if len(pred_class_ids) != 0:  # 上面算的AP是所有类的AP，现在加一个每一类计算AP值的。
        for c in classes:  # 对于每一类循环
            this_class_id = np.where(pred_class_ids == c)[0]
            pred_boxes_this_class = pred_boxes[this_class_id]  # 被认为是这一个类别的检测结果的外接矩形
            this_class_id_gt = np.where(gt_class_ids == c)[0]
            gt_boxes_this_class = gt_boxes[this_class_id_gt]  # 这一类别的所有物体的金标准外接矩形
            overlaps_this_class = compute_overlaps(pred_boxes_this_class, gt_boxes_this_class)

            match_count_this_class = 0  # 这一个类别的、判断对了的个数。现在初始化为0。
            pred_match_this_class = np.zeros([pred_boxes_this_class.shape[0]])  # “被认为是这一类的、预测外接矩形个数”个0。
            gt_match_this_class = np.zeros([gt_boxes_this_class.shape[0]])  # “这一类别金标准外接矩形个数”个0。
            for i in range(len(pred_boxes_this_class)):  # i是对每个预测外接矩形循环
                sorted_ixs_this_class = np.argsort(overlaps_this_class[i])[::-1]  # 类似于前面，只不过都改成这一类的。
                for j in sorted_ixs_this_class:
                    # If ground truth box is already matched, go to next one
                    if gt_match_this_class[j] == 1:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop  如果碰上的iou小于阈值，停止循环。
                    iou_this_class = overlaps_this_class[i, j]
                    if iou_this_class >= iou_threshold:  # 这儿因为是对当前类别做的，所以类别肯定是对的了。所以只要IoU大于这个阈值，就可以认为匹配上了。
                        match_count_this_class += 1  # 匹配个数+1。
                        gt_match_this_class[j] = 1
                        pred_match_this_class[i] = 1  # 当前预测外接矩形置1。
                    else:
                        break  # 退出j的循环，直接弄下一个预测外接矩形（下一个i）。

            precisions_this_class = np.cumsum(pred_match_this_class) / (np.arange(len(pred_match_this_class)) + 1)  # 当前类的准确率和召回率。
            recalls_this_class = np.cumsum(pred_match_this_class).astype(np.float32) / len(gt_match_this_class)
            precisions_this_class = np.concatenate([[0], precisions_this_class, [0]])  # 在当前类准确率前后各添一个0。
            recalls_this_class = np.concatenate([[0], recalls_this_class, [1]])  # 在当前类召回率前面添一个0，后面添一个1。
            for i in range(len(precisions_this_class) - 2, -1, -1):
                precisions_this_class[i] = np.maximum(precisions_this_class[i], precisions_this_class[i + 1])

            # Compute mean AP over recall range  计算召回率范围内的平均AP。
            indices_this_class = np.where(recalls_this_class[:-1] != recalls_this_class[1:])[0] + 1
            AP_this_class = np.sum((recalls_this_class[indices_this_class] - recalls_this_class[indices_this_class - 1]) * precisions_this_class[indices_this_class])
            APc.append(AP_this_class)

    return mAP, precisions, recalls, overlaps, classification, overlap_mean, APc


# def compute_ap1(num_classes, gt_boxes, gt_class_ids, gt_masks,
#                 pred_boxes, pred_class_ids, pred_scores, pred_masks,
#                 iou_threshold=0.5):
#     """Compute Average Precision at a set IoU threshold (default 0.5).
#     Returns:
#     mAP: Mean Average Precision
#     precisions: List of precisions at different class score thresholds.
#     recalls: List of recall values at different class score thresholds.
#     overlaps: [pred_masks, gt_masks] IoU overlaps.
#     """
#     # Get matches and overlaps
#     gt_match, pred_match, overlaps, match_count = compute_matches(
#         gt_boxes, gt_class_ids, gt_masks,
#         pred_boxes, pred_class_ids, pred_scores, pred_masks,
#         iou_threshold)
#
#     # Compute precision and recall at each prediction box step
#     precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
#     recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
#
#     # Pad with start and end values to simplify the math
#     precisions = np.concatenate([[0], precisions, [0]])
#     recalls = np.concatenate([[0], recalls, [1]])
#
#     # Ensure precision values decrease but don't increase. This way, the
#     # precision value at each recall threshold is the maximum it can be
#     # for all following recall thresholds, as specified by the VOC paper.
#     for i in range(len(precisions) - 2, -1, -1):
#         precisions[i] = np.maximum(precisions[i], precisions[i + 1])
#
#     # Compute mean AP over recall range
#     indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
#     mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
#                  precisions[indices])
#     classes = (np.arange(num_classes - 1) + 1)
#     if len(pred_class_ids) != 0:
#         # compute overall classification accuracy rate
#         classification = match_count / len(pred_class_ids)
#         APc = []
#         for c in range(len(classes)):
#             pred_match_count = 0
#             for i in range(len(pred_class_ids)):
#                 if pred_class_ids[i] == classes[c]:
#                     pred_match_count += 1
#
#             gt_match_count = 0
#             for j in range(len(gt_class_ids)):
#                 if gt_class_ids[j] == classes[c]:
#                     gt_match_count += 1
#             if gt_match_count != 0:
#                 # compute TP/(TP + FP)
#                 if pred_match_count > gt_match_count:
#                     APclasses = gt_match_count / pred_match_count
#                 else:
#                     APclasses = pred_match_count / gt_match_count
#             else:
#                 APclasses = 0
#             APc.append(APclasses)
#     else:
#         classification = 0
#         APc = []
#         for c in range(len(classes)):
#             APclasses = 0.0
#             APc.append(APclasses)
#     # The correct index of the circumscribed rectangle (refers to the index number in the predicted circumscribed rectangle)
#     match = np.where(pred_match == 1)
#     overlap_match = overlaps[match, :]
#     overlap_for_each_propasal = np.max(overlap_match, axis=2)
#     overlap_mean = np.average(overlap_for_each_propasal)
#
#     return mAP, precisions, recalls, overlaps, APc, classification, overlap_mean

def select_high_scores_for_mp(input_detections, scores, threshold, num_classes):
    """和前面的一样，只不过把那个ix也选出来排序、返回。测试一下多个返回的函数怎么搞。"""
    ix = tf.where(scores > threshold)[:, 0]
    selected = tf.gather(input_detections, ix)
    first_row = selected[:,0]  # 第0列，即所有探测结果的y1坐标
    column_num = tf.size(first_row)  # 第0列里有几个数？
    descending_index = tf.nn.top_k(-first_row, column_num)[1]  # 按照选出来的这一列从小到大排序（负号）
    sorted = tf.gather(selected, descending_index)  # 按照第0列排序

    max_class_id = tf.cast(tf.reduce_max(sorted[:,4], axis=0), dtype=tf.int32)  # axis=0是这一列所有数中最大的，变成tf.int32。
    min_class_id = tf.cast(tf.reduce_min(sorted[:,4], axis=0), dtype=tf.int32)  # 这一列所有数中最小的
    P_before = tf.cast((num_classes - 1 - max_class_id), dtype=tf.int32)
    P_after = tf.cast(min_class_id - 1, dtype=tf.int32)
    # padded = np.pad(sorted, ((P_before, P_after), (0, 0)),'constant',constant_values = (0))
    padded = tf.pad(sorted, ((0, P_before+P_after), (0, 0)),'constant',constant_values = (0))
    # 上句，上面补P_before个0，下面补P_after个0。后来改成，把这些0都补在后面，这样便于后面弄rnn中的mask。
    # <tf.Tensor 'Pad:0' shape=(?, ?) dtype=float32>
    # <tf.Tensor 'Pad_1:0' shape=(?, ?) dtype=float32>
    # padded_ix = np.pad(sorted_ix, ((P_before, P_after)), 'constant', constant_values=(-1))
    return padded