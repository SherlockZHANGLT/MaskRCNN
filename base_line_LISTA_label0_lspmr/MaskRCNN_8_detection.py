#coding=utf-8
import tensorflow as tf
import MRCNN_utils
def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final detections.
    处理一张图的提出。细化那些提出，并且过滤掉重合的，返回最终探测结果。
    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            输入的rois，即外面传来的rois，shape是(?, ?)，两个?应该分别是N和4，分别是每张图中补零后的提出的数量（就是那个1000）、还有那4个角点坐标。
        probs: [N, num_classes]. Class probabilities.
            输入probs，即外面传来的mrcnn_class，shape是(1000, 7)，分别是补零后的提出数、类别总数。
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific bounding box deltas.
            输入的deltas，即外面传来的mrcnn_bbox，这里输入的是每一张图的外接矩形修正值。shape=(1000, 7, 4)，
            表示有1000个提出，每个提出都有可能是7种类别，然后每个类别都有4个角点。
        window: (y1, x1, y2, x2) in image coordinates. The part of the image that contains the image excluding the padding.
            输入的windows，图像窗口。
    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    【注意】_detections和_mrcnn_bbox的关系：
        是把那个输入的rpn_rois用mrcnn_bbox修正，就得到了_detections的前4个坐标值，这个是在DetectionLayer函数里搞定的。
        如果训练得比较好了，会发现_rpn_rois和_detections的非零部分是比较接近的，因为那个修正值应该不会太大。
        然后，这个detections的前4个数就是预测的外接矩形的归一化了的坐标，如果乘以512，基本上就是预测的外接矩形了.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)  # shape=(1000,)，每个提出的类别
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    """上句indices的shape是(1000, 2)，上一行1000个数是从0~999，下一行1000个数是这1000个提出的类别序号。
    在MaskRCNN_7_losses.mrcnn_bbox_loss_graph函数里也有这种把索引号和类别对起来拼成(?, 2)的矩阵的做法。
    """
    class_scores = tf.gather_nd(probs, indices)  #
    """上句执行后shape是(1000,)，“每个提出 被判断为相应的class_ids所指代的 类别”的概率。测试模式下执行，上句的输出是：class_scores：<tf.Tensor 'mask_rcnn_model/GatherNd_4:0' shape=(100,) dtype=float32>"""
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    """上句执行后shape是(1000, 4)，注意输入的deltas的shape是(1000, 7, 4)，所以这也是对每个提出，找到这一类的外接矩形。
    测试模式下执行，上句的输入和输出分别是：
    deltas：<tf.Tensor 'mask_rcnn_model/strided_slice_151:0' shape=(100, 4, 4) dtype=float32>
    indices：<tf.Tensor 'mask_rcnn_model/stack_6:0' shape=(100, 2) dtype=int32>
    deltas_specific：<tf.Tensor 'mask_rcnn_model/GatherNd_5:0' shape=(100, 4) dtype=float32>，或者把5换成7。
    """
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = MRCNN_utils.apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    """用选好了类别的外接矩形修正值deltas_specific，去修正rois。config.BBOX_STD_DEV是array([0.1, 0.1, 0.2, 0.2])。
    也就是说，这儿是从 提出的角点坐标 和 外接矩形修正值，变成最终外接矩形的坐标。
    测试模式下执行，上句的输入和输出分别是：
    rois：<tf.Tensor 'mask_rcnn_model/strided_slice_149:0' shape=(?, ?) dtype=float32>
    deltas_specific：<tf.Tensor 'mask_rcnn_model/GatherNd_5:0' shape=(100, 4) dtype=float32>
    refined_rois：<tf.Tensor 'mask_rcnn_model/apply_box_deltas_out_2:0' shape=(100, 4) dtype=float32>，或者把2换成3。
    """
    # Clip boxes to image window
    refined_rois = MRCNN_utils.clip_boxes_graph(refined_rois, window)
    """
    测试模式下执行，上句的输出是：
    refined_rois：<tf.Tensor 'mask_rcnn_model/clipped_boxes_2:0' shape=(100, 4) dtype=float32>，或者把2换成3。
    """
    # TODO: Filter out boxes with zero area
    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    """上句，找到类别序号不为0的地方，即要去掉补零（补零部分的class_ids都是0），而且把背景类给删掉了。
    现在shape=(?,)，所以keep是类别序号不为0的地方在class_ids（那1000个位置）中的位置索引。
    比如说，keep=[0, 1, 2, 4, 6, 7, 8, ...]，keep里面的东西，应该是选中的提出在原来那1000个提出里的索引号。
    """
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]  # 找到得分大于那个阈值的地方，即滤掉分值太小的提出。shape=(?,)，所以conf_keep是类别分数大于阈值的地方在class_scores（也是有1000个位置）中的位置索引。
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        """上句是求交集，两个东西expand_dims后的shape都是(1,?)，?代表满足条件的提出个数。keep经过了这一步和下一步后，shape是(?,)，?应该是代表同时满足两个条件的提出个数。"""
        keep = tf.sparse_tensor_to_dense(keep)[0]
    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)  # 选出来keep（类别序号不为0且得分大于阈值的提出，在那1000个提出中的位置索引）对应的那几个提出的类别。shape=(?,)。
    pre_nms_scores = tf.gather(class_scores, keep)  # 选出来keep对应的那几个提出的得分。shape=(?,)。
    pre_nms_rois = tf.gather(refined_rois,   keep)  # 选出来keep对应的那几个提出的细化了的外接矩形（不再是修正值了）。shape=(?, 4)。
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]  # 去掉重复的，因为pre_nms_class_ids是不同提出的类别号，肯定会有重复的。可以想象在我们这儿unique_pre_nms_class_ids应该就是1~6一样一个。shape=(?,)。
    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        """上句，找到属于某一类的所有物体的索引号。shape=(?,)。注意输入的class_idshape=()。
        后面，如果某两个属于某一类的提出不怎么重合，那就不管（比如说，就认为是两块椎骨或者两个人脸）；
        如果某两个属于某一类的提出重合较多，那就认为可能这两个提出找到了同一个物体，那就用tf.image.non_max_suppression删掉得分比较小的那个。
        """
        # Apply NMS
        class_keep = tf.image.non_max_suppression(tf.gather(pre_nms_rois, ixs), tf.gather(pre_nms_scores, ixs),
                                                  max_output_size=config.DETECTION_MAX_INSTANCES,
                                                  iou_threshold=config.DETECTION_NMS_THRESHOLD)#
        """上句在通过pre_nms_scores，把pre_nms_rois做了非最大抑制，主要解决的是一个目标被多次检测的问题。shape=(?,)。
        那个config.DETECTION_MAX_INSTANCES是100，就是说最多保留100个提出。config.DETECTION_NMS_THRESHOLD是说如果两个提出重合度超过这个阈值，就删掉一个。
        想起来了，所以我们一般看不到同一类别的两个外接矩形重合很大，就像下面网址中的那个人脸，那么多提出框重合在一起了：
            https://blog.csdn.net/wuguangbin1230/article/details/79895364
        但是这个似乎不能解决不同类别的东西重合的问题。。
        """
        # Map indicies
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        """上句，class_keep里应该是想要的提出的原始索引号（而不是在那个keep里的索引号），比如，0, 1, 2, 4, 6, 7。shape=(?,)。"""
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)  # shape=(?,)。
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])  # shape=(100,)
        return class_keep
    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    """
    上句，输出的nms_keep的shape是shape=(?, 100)。这个?可能是6，因为unique_pre_nms_class_ids应该是有1~6这6个类（注意到这儿输入的时候shape是）(?,)，
        然后在那个nms_keep_map函数里面，就成了()了，这是说明应该是执行了6次nms_keep_map函数，每次处理unique_pre_nms_class_ids中的1个。
        然后每个类都返回了一个class_keep，即那100个索引号，现在就有好几个100个索引号堆叠起来，所以就是shape=(?, 100)。
    【基础】tf.map_fn函数
    似乎第一个参数是一个函数，第二个参数是这个函数的输入参数（可以是由若干个输入参数组成的一个tensor，
        比如说那个函数的输入是一个数，而现在这儿可以输入一个由N个数组成的向量），返回的就是N个“这个函数的返回值”堆叠起来的张量。
    所以这个东西用来处理那些不知道长度的张量，似乎是个好方法啊，因为如果想用for i in range()的话，range()里的东西必须是确定的，如果是个?就会报错。
        因为如果要把不同器官的掩膜用tf合并的话，那就不知道多少个掩膜，想直接做估计是不行，但我可以用这个，每个掩膜都返回一个放缩后的(512, 512)的图，
        然后tf.map_fn的结果就是(?, 512, 512)的了。。
    """
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])  # 执行后shape=(?,)
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])  # 执行后shape=(?,)，把补的-1都去掉了。
    """上面，reshape就是把所有类别的100个提出，给拼在一起。然后去掉补的-1。"""
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))  # 执行后shape=(?,)
    keep = tf.sparse_tensor_to_dense(keep)[0]  # 执行后shape=(?,)
    """正式施行非最大化抑制，选出来要保留的索引号。"""
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES  # 就是那个100。
    class_scores_keep = tf.gather(class_scores, keep)  # 执行后shape=(?,)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)  # 执行后shape=()，最终要保留的提出数。
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]  # 执行后shape=(?,)
    keep = tf.gather(keep, top_ids)
    """以上，再找出来得分最大的那100个来。
    测试模式下执行，上句的输出是：
    keep：
    """
    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)  # shape=(?, 6)，但tf.shape(detections)[0]就直接是shape=()了。
    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

def DetectionLayer(rois, mrcnn_class, mrcnn_bbox, image_meta, config):
    print(rois, mrcnn_class, mrcnn_bbox, image_meta)
    _, _, window, _ = MRCNN_utils.parse_image_meta_graph(image_meta)
    window = tf.cast(window, tf.float32)
    detections_batch = MRCNN_utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
        lambda x, y, w, z: refine_detections_graph(x, y, w, z, config), config.IMAGES_PER_GPU)
    """
    上句是先把rois, mrcnn_class, mrcnn_bbox等输入refine_detections_graph函数，得到每一张图的detections，补零，再用batch_slice弄成一个批次。
    注意到，输入的mrcnn_bbox很明显是外接矩形修正值；而输出的detections_batch中，每一行的6个数中，第0~3个数是外接矩形的角点坐标值（此处应该是归一化了的）。
    输出detections_batch是<tf.Tensor 'mask_rcnn_model/packed_3:0' shape=(2, ?, ?) dtype=float32>
    """
    results = tf.reshape(detections_batch, [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 6])  # shape=(批大小, 100, 6)
    """
    返回的东西，会是个三维张量，第0维是对应不同的图像，
    然后每个给定的第0维，应该都是一个二维矩阵，每一行有6个数，有config.DETECTION_MAX_INSTANCES行。
    前面几行是有数的（后面是补的零），第0~3个数是（归一化了的）外接矩形角点坐标（这次不是修正值了），第4个是类别，第5个是得分。
    """
    return results