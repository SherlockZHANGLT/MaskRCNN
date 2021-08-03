#coding=utf-8
import tensorflow as tf
import MRCNN_utils

def DetectionTargetLayer(rpn_rois, rpn_scores_nms, corresponding_anchors, input_gt_class_ids, gt_boxes, input_gt_masks, input_sparse_gt, config):
    names = ["rois", "rpn_scores_selected", "corresponding_anchors_selected",
             "target_class_ids", "target_bbox", "target_mask", "target_sparse", "positive_rois"]
    # 注意这个names的个数，必须要和detection_targets_graph输出的东西的个数相同！否则会报错！！
    outputs = MRCNN_utils.batch_slice(
        [rpn_rois, rpn_scores_nms, corresponding_anchors, input_gt_class_ids, gt_boxes, input_gt_masks, input_sparse_gt],
        lambda w, x, y, z, a, b, c: detection_targets_graph(w, x, y, z, a, b, c, config),
        config.IMAGES_PER_GPU, names=names)  # 用的那个detection_targets_graph函数似乎也可以放在下面？。。
    """
    分析一下上句：
    首先这一句是必须用这个batch_slice函数的，因为那个detection_targets_graph只能处理一张图，但实际上我们有batch_size张图。
    然后具体地：
    MRCNN_utils.batch_slice的4个输入分别是inputs, graph_fn, batch_size, names
    inputs是[rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks]，而不是那个detection_targets_graph的输出。
    graph_fn是那个lambda函数。但是这个函数的输出放到哪儿去了？应该就是batch_slice函数里的output_slice。
    batch_size是那个config.IMAGES_PER_GPU
    然后开始执行程序，结果发现：
        在那个batch_slice函数里打印出来inputs，是：
            [<tf.Tensor 'packed_2:0' shape=(2, ?, ?) dtype=float32>,  原来的shape是(4,?,?)，现在(2,?,?)应该是一样的，只是批次中的图像张数变了。
             <tf.Tensor 'input_gt_class_ids:0' shape=(2, ?) dtype=int32>,  没细看，但目测这个shape和input_gt_class_ids的shape应该一样。
             <tf.Tensor 'truediv:0' shape=(2, ?, 4) dtype=float32>,  同上
             <tf.Tensor 'input_gt_masks:0' shape=(2, 56, 56, ?) dtype=bool>]  同上
        然后，在那个batch_slice函数里打印出来inputs_slice，是：
            [<tf.Tensor 'strided_slice_69:0' shape=(?, ?) dtype=float32>,
             <tf.Tensor 'strided_slice_70:0' shape=(?,) dtype=int32>,
             <tf.Tensor 'strided_slice_71:0' shape=(?, 4) dtype=float32>,
             <tf.Tensor 'strided_slice_72:0' shape=(56, 56, ?) dtype=bool>]
             发现这些就是一个批次里的2张图中取了1张，然后还是rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks那四个输入。
             反正有个for循环，所以是对这一批中的所有图都做了的。
        再后，output_slice = graph_fn(*inputs_slice)得到的output_slice是：
            (<tf.Tensor 'Pad_3:0' shape=(?, ?) dtype=float32>,
             <tf.Tensor 'Pad_5:0' shape=(?,) dtype=int32>,
             <tf.Tensor 'Pad_6:0' shape=(?, ?) dtype=float32>,
             <tf.Tensor 'Pad_7:0' shape=(?, ?, ?) dtype=float32>)
        再后那个if not isinstance...是把output_slice变成list，然后append到outputs里去。
        所以，for循环执行1次，那个outputs就多了一个output_slice。那么，for循环执行完毕后，outputs是：
            [(<tf.Tensor 'Pad_3:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_5:0' shape=(?,) dtype=int32>,
              <tf.Tensor 'Pad_6:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_7:0' shape=(?, ?, ?) dtype=float32>),
             (<tf.Tensor 'Pad_8:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_10:0' shape=(?,) dtype=int32>,
              <tf.Tensor 'Pad_11:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_12:0' shape=(?, ?, ?) dtype=float32>)]
        然后那个list(zip(*outputs))是给他重排了一下：
            [(<tf.Tensor 'Pad_3:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_8:0' shape=(?, ?) dtype=float32>),
             (<tf.Tensor 'Pad_5:0' shape=(?,) dtype=int32>,
              <tf.Tensor 'Pad_10:0' shape=(?,) dtype=int32>),
             (<tf.Tensor 'Pad_6:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_11:0' shape=(?, ?) dtype=float32>),
             (<tf.Tensor 'Pad_7:0' shape=(?, ?, ?) dtype=float32>,
              <tf.Tensor 'Pad_12:0' shape=(?, ?, ?) dtype=float32>)]
        然后result = [tf.stack...]是给他堆叠起来，把2张图变成1个batch的那种。输出是：
            [<tf.Tensor 'rois:0' shape=(2, ?, ?) dtype=float32>,
             <tf.Tensor 'target_class_ids:0' shape=(2, ?) dtype=int32>,
             <tf.Tensor 'target_bbox:0' shape=(2, ?, ?) dtype=float32>,
             <tf.Tensor 'target_mask:0' shape=(2, ?, ?, ?) dtype=float32>]
        然后，上句的outputs也就是那个result，一模一样的。
        所以我就不明白了，我detection_targets_graph就多弄一个输出有什么不对的么？【已解决，见下】
            发现，如果detection_targets_graph有5个输出，也可以执行batch_slice函数，得到输出outputs，但是——
            outputs里仍然只有4个东西，即也是这个：
            [<tf.Tensor 'rois:0' shape=(2, ?, ?) dtype=float32>,
             <tf.Tensor 'target_class_ids:0' shape=(2, ?) dtype=int32>,
             <tf.Tensor 'target_bbox:0' shape=(2, ?, ?) dtype=float32>,
             <tf.Tensor 'target_mask:0' shape=(2, ?, ?, ?) dtype=float32>]
            这样，返回去一个只有4个的东西，而外面是有rois, target_class_ids, target_bbox, target_mask, proposals这5个，所以就不对了。
            报错是“ValueError: not enough values to unpack (expected 5, got 4)”。
        然后看了一下原因，这是因为batch_slice函数里面有一个for o, n in zip(outputs, names)，
            如果detection_targets_graph有5个输出，那么batch_slice函数里的那个outputs是没问题的，是这样的东西：
            [(<tf.Tensor 'Pad_18:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_23:0' shape=(?, ?) dtype=float32>),
             (<tf.Tensor 'Pad_20:0' shape=(?,) dtype=int32>,
              <tf.Tensor 'Pad_25:0' shape=(?,) dtype=int32>),
             (<tf.Tensor 'Pad_21:0' shape=(?, ?) dtype=float32>,
              <tf.Tensor 'Pad_26:0' shape=(?, ?) dtype=float32>),
             (<tf.Tensor 'Pad_22:0' shape=(?, ?, ?) dtype=float32>,
              <tf.Tensor 'Pad_27:0' shape=(?, ?, ?) dtype=float32>),
             (<tf.Tensor 'trim_proposals_3/Gather:0' shape=(?, ?) dtype=float32>,【这个和下一行的就对应多输出的那个东西啊】
              <tf.Tensor 'trim_proposals_4/Gather:0' shape=(?, ?) dtype=float32>)]
            但是，那个names里面只有4个啊，所以最后那俩trim_proposals_3/Gather:0就没循环到啊。。
        所以，在names里加上了"proposals", "gt_boxes", "overlaps"这三个，对应detection_targets_graph新加的三个输出。
        然而又有个问题，如果一批中的两张图的gt_boxes个数不一样，那么tf.stack就会出错（InvalidArgumentError : Shapes of all inputs must match: 
            values[0].shape = [1151,30] != values[1].shape = [1059,30]这样的报错）
            （事实上，发现他原来返回的前4个变量，都是补零的，补零后长度就一样了。这也是补零的作用！）。
    """
    return outputs

def detection_targets_graph(proposals, rpn_scores_nms, corresponding_anchors, gt_class_ids, gt_boxes, gt_masks, gt_sparse, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.
    给一张图生成探测目标：下采样的提出、目标类别ID、外接矩形修正、掩膜。

    Inputs:
    输入：类似于“class DetectionTargetLayer”的输入情况，但第一维的batch没了，因为现在只处理一张图。
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
        这个proposals应该是一个的矩阵，表示一张图中的N个提出的外接矩形的角点坐标。
            如果实际上RPN生成的提出数量没到N，就把这个矩阵后面的东西都补零。
        有点奇怪的是，他输入进来的shape还是(?,?)，可能是因为上面【第四步】输出的shape是(4,?,?)，但如其中分析，就理解为(N,4)好了。
        【基础】看了后面程序应该明白了，前面那个?是因为传进来的时候是N，等到去掉补零后就变成了V_p，在变的，所以是个?。
            后面的那个?应该是因为，我们知道这个张量是存角点的，但是电脑不知道，所以是?，得等到feed进来之后，才会知道的。。
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        这个gt_class_ids应该是一个长度为MAX_GT_INSTANCES的向量，表示这MAX_GT_INSTANCES个金标准提出的类别。
            同样地，如果实际上金标准提出数没到MAX_GT_INSTANCES，也会补零。
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        类似上面，是一个[MAX_GT_INSTANCES,4]的矩阵，表示一张图中的MAX_GT_INSTANCES个金标准提出的外接矩形。
            如果实际上金标准提出数没到MAX_GT_INSTANCES，也会补零。补零数应该和上面gt_class_ids里的0数是一样的。
            输入的shape是(?,4)
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    返回：类似于“class DetectionTargetLayer”，返回目标ROI，对应的类别ID、外接矩形位移、掩膜。第一维的batch也没了。
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinments.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)
        # Remove zero padding
    proposals, _ = MRCNN_utils.trim_zeros_graph(proposals, name="trim_proposals")
    # 上句，从proposals去掉补了的零。shape=(?,?)，理解为[有值的提出数V_p,4]，注意那个V_p<N。
    gt_boxes, non_zeros = MRCNN_utils.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    # 上句，从gt_boxes中去掉补了的0，同时non_zeros是“有值的地方为1，补零的地方为0”的那个矩阵。
    #     注意到提出有个外接矩形，这儿还有个金标准外接矩形。金标准也有可能没有MAX_GT_INSTANCES个物体，所以要补零。现在去掉他们。
    #     执行后，gtbox的shape=(?,4)，理解为[有值的金标准提出数V_g,4];
    #         non_zeros的shape=(?,)，理解为[有值的金标准提出数V_g,]，
    #         也就是一个向量，里面有V_g个元素，每个元素应该就是对应一个金标准提出的索引（在输入的[MAX_GT_INSTANCES,4]的矩阵中的索引号）。
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    # 上句，根据金标准外接矩形的有值的地方，选出来金标准类别ID的有值的地方。
    #     注意到用的是同一个non_zeros，这其实是显然的，因为金标准提出里有值的个数是固定的啊。
    #     执行后，gt_class_ids的shape=(?,)，理解为[有值的金标准提出数V_g,]，
    #     里面有V_g个元素，每个元素应该就是对应一个金标准提出的类别（所谓class_id）。
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")
    # 上句，应该也是金标准外接矩形的有值的地方，选出来金标准掩膜的有值的地方。
    #     执行后，shape是(56,56,?)。最后这个?应该也对应这个V_g，因为每个有值的金标准提出都应该对应一个掩膜啊。
    #     （后面会用permute把这个?换到最前面去）
    gt_sparse = tf.boolean_mask(gt_sparse, non_zeros, name="trim_gt_sparse")
    """以上，处理了RPN提出和金标准提出编码后的矩阵，把补的那些零删掉了。
    【基础】补零
    一种处理、选择提出的方法。
    先都补零，这样对于任意一张图，输入进来的proposals就都是一样的了。然后
        用gather函数，选出来所需要的那些提出（删除补零、剔除拥挤矩形、选取部分正例负例即下采样）。
        注意到，无论是proposals还是gt_boxes的形状都在变化，所以就一直是?了。
    【基础】删除补零
    即，从张量中删除若干行。可以用tf.boolean_mask函数，它的第一个输入为原来的张量，第二个参数是和原来张量长度相等的、由True和False组成的向量。
        如，1维的情况：tensor = [0, 1, 2, 3]  shape是(4,)
                     mask = np.array([True, False, True, False])  这个长度就是4。
                     boolean_mask(tensor, mask)  # [0, 2]
            2维的情况：tensor = [[1, 2], [3, 4], [5, 6]]  shape是(3, 2)
                      mask = np.array([True, False, True])  这个长度就是3
                      boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
    具体见25(choose_from_tensor).py。
    """

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]  # shape=(?,)此处?是拥挤矩形个数，记作V_c。
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]  # shape=(?,)此处?是V_g-V_c。
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)  # shape=(?,4)此处?是V_c。
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)  # shape=(?,)此处?是V_g-V_c。
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)  # 此处shape仍然是(?,4)，但?不再是刚才的那个V_g，而是V_g-V_c。因为现在已经去掉了拥挤矩形了。
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)  # 同上，此处shape仍然是(56,56,?)，但?是V_g-V_c了。
    gt_sparse = tf.gather(gt_sparse, non_crowd_ix)  # 类似于gt_boxes的处理
    """以上，处理“拥挤的外接矩形”，拿出非拥挤矩形的金标准类别ID、外接矩形、掩膜。注意，这些都是对金标准做的，
    也就是说，是把金标准中的拥挤矩形删除掉，跟当前做的结果没有关系！！
    从程序看来，这些拥挤矩形的、为负数的类别ID，在此之前就已经弄出来了。因为这直接用的是gt_class_ids < 0啊，
        说明这个时候，这个gt_class_ids都是已知的了（事实上，那个data_generator里的load_image_gt函数里，
        弄出来了这个金标准类别，即gt_class_ids。然后它作为模型输入，输入到了class MaskRCNN这个模型里）。
    """

    overlaps = MRCNN_utils.overlaps_graph(proposals, gt_boxes)
    # 上句，第一次执行后是<tf.Tensor 'mask_rcnn_model/Reshape_1:0' shape=(?, ?) dtype=float32>，两个?应该分别是V_p（提出数）和
    #     V_g-V_c（金标准中的非拥挤矩形数），然后这个overlap矩阵中的一行，应该是某一个提出和所有金标准外接矩阵的IOU。
    # 第二次执行是 <tf.Tensor 'mask_rcnn_model/Reshape_5:0' shape=(?, ?) dtype=float32>
    # Compute overlaps with crowd boxes [anchors, crowds]   计算提出和拥挤的外接矩形的重合
    crowd_overlaps = MRCNN_utils.overlaps_graph(proposals, crowd_boxes)  #
    # 上句，也是计算提出和某些矩形的重合度，只不过这个是那些“拥挤矩形”。拥挤矩形也属于金标准提出的一部分，但后面应该是要排除。
    #     计算出来shape=(?,?)，这两个?应该分别是V_p和V_c。理解同上。
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    # shape=(?,)，就是每一个提出，和拥挤矩形的最大重合度。这个?应该是V_p，即V_p个数，每个数都表示一个提出和所有拥挤矩形的最大重合度。
    no_crowd_bool = (crowd_iou_max < 0.001)
    """
    以上，对所有的提出，计算了：1、他们和金标准外接矩形的重合度；2、他们和拥挤的外接矩形的重合度（并以此找出no_crowd_bool）。
    重合度overlaps是个矩阵，其中的一行，应该是某一个提出和所有金标准外接矩阵的IOU。

    注意！锚和提出的区别：提出是细化了的锚。这个detection_targets_graph函数是在构造class MaskRCNN的【从图片到提出的第五步】执行的，
        而这之前的【从图片到提出的第四步】正是把锚做了选择、细化、非最大抑制等等操作，所以此处锚和提出是不一样的。
        而，在那个data_generator函数中，是把锚直接拿来和金标准外接矩形对比的。
    """

    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 上句，第一次执行后<tf.Tensor 'mask_rcnn_model/Max_7:0' shape=(?,) dtype=float32>，就是每一个提出和金标准外接矩形（已经去除
    #     了拥挤矩形的）的最大重合度。类似于上面，?应该是V_p，提出数。
    # 第二次执行是<tf.Tensor 'mask_rcnn_model/Max_9:0' shape=(?,) dtype=float32>
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    # 上句，执行后shape=(?,)，此时?应该是有值、且和金标准外接矩形重合度大于0.5的提出数（应该是比V_p还小，记作V_pp吧）
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 上句，执行后shape=(?,)，?应该是V_pp，这个向量就表示V_pp个正例的ID吧。
    # 执行完后，positive_indices是<tf.Tensor 'mask_rcnn_model/strided_slice_87:0' shape=(?,) dtype=int64>
    #     再来一次，就是 <tf.Tensor 'mask_rcnn_model/strided_slice_130:0' shape=(?,) dtype=int64>
    if config.ONE_INSTANCE_EACH_CLASS:
        each_gt_max_pos = tf.argmax(overlaps, axis=0)  # ##########改的，找出“和每个金标准提出的重合度最大的提出”的索引号。##########
        # 第一次执行是<tf.Tensor 'mask_rcnn_model/ArgMax:0' shape=(?,) dtype=int64>
        # 第二次是<tf.Tensor 'mask_rcnn_model/ArgMax_2:0' shape=(?,) dtype=int64>
        c = tf.sets.set_intersection(positive_indices[None, :], each_gt_max_pos[None, :])  # ##########改的，和每个金标准重合度最大，且重合度大于0.5的索引号。##########
        # 其实就是一个求交集的过程。
        positive_indices = c.values  # ##########改的，更新一下正例索引号，即，重合度大于0.5，且和每个金标准重合度最大。##########
        # 这样，在训练好了的情况下，这个positive_indices里就应该只有3个（就是非背景类别数，本项目是3个，因为除了背景外有L4/L5/S1三类）了。
        #     但是，在没训练好的情况下，可能是0，因为一开始可能所有的提出的重合度都不大于0.5的。
        # 上句执行完后，positive_indices是<tf.Tensor 'mask_rcnn_model/DenseToDenseSetOperation:1' shape=(?,) dtype=int64>
        # 然后第二次来，就是<tf.Tensor 'mask_rcnn_model/DenseToDenseSetOperation_2:1' shape=(?,) dtype=int64>
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]  # shape=(?,)
    # 以上找到反例：和所有的金标准外接矩形重合度都小于0.5，且不是拥挤矩形的，就叫做反例。
    # config.ONE_INSTANCE_EACH_CLASS=True的时候，上句执行完后，negative_indices是<tf.Tensor 'mask_rcnn_model/strided_slice_90:0' shape=(?,) dtype=int64>
    """以上一段，是根据各个提出和金标准外接矩形的重合度，从那些提出（理解为：提出是挑选出来的一部分锚）中提取出来所有的正例和负例。
    注意到，这个地方算over拉PS的时候，用的proposals是第四步推断出来的那些提出，来计算提出和金标准外接矩形的IOU的，
        所以后面要做的，其实就是给上一步的所有提出，都给“分配”了一个离他最近的（也就是重合最大的）金标准。
    """
    if not config.ONE_INSTANCE_EACH_CLASS:
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)  ##########改的，仍然不选正例了，所有正例都用上，因为一共就3个。##########
        # 上句，正例数。这是：每张图片上的ROI数（在organs_trainings里设成是150）*正例ROI所占的比例（在config.py里设成是0.33）##########
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]  # ##########改的##########
        # shape=(?,)，此处?应该是positive_count，即从那V_pp个正例中随机取出来了positive_count个。##########
    positive_count = tf.shape(positive_indices)[0]  # 应该就是7~9个（非背景类别数个）了。
    # 上句执行后，shape=()，变成一个数了。原来也是一个数，但是在电脑里是张量的形式吧。
    # config.ONE_INSTANCE_EACH_CLASS=True的时候，跑完后是<tf.Tensor 'mask_rcnn_model/strided_slice_91:0' shape=() dtype=int32>
    # 然后第二次来<tf.Tensor 'mask_rcnn_model/strided_slice_134:0' shape=() dtype=int32>
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count  # shape=()，应该就是6个了。。。
    # 这样的话，按理说可以不补零了，因为一共就9个啊。如果把那个TRAIN_ROIS_PER_IMAGE设成((NUM_CLASSES - 1) / ROI_POSITIVE_RATIO)
    # config.ONE_INSTANCE_EACH_CLASS=True的时候，跑完后是<tf.Tensor 'mask_rcnn_model/sub_36:0' shape=() dtype=int32>
    # 然后第二次来<tf.Tensor 'mask_rcnn_model/sub_65:0' shape=() dtype=int32>
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]  # shape=(?,)，和positive_indices类似。
    # 上面三句类似，从负例索引号里随机取出来negative_count个负例。这个还是得要的。。
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    # 上句，第一次执行后<tf.Tensor 'mask_rcnn_model/Gather_15:0' shape=(?, ?) dtype=float32>，表示这些选定了的正例提出（就是外接矩形）。
    #     第二次是<tf.Tensor 'mask_rcnn_model/Gather_29:0' shape=(?, ?) dtype=float32>
    #     第一个?应该就是positive_count，表示选定了这么多个正例；第二个?应该是4，表示4个角点。
    negative_rois = tf.gather(proposals, negative_indices)
    # 上两句是选定了的把正例和负例的索引号提出来。<tf.Tensor 'mask_rcnn_model/Gather_16:0' shape=(?, ?) dtype=float32>
    # 第二次是<tf.Tensor 'mask_rcnn_model/Gather_30:0' shape=(?, ?) dtype=float32>
    """以上，下采样ROI，其实就是随机地选择指定个数的正例和负例。"""
    positive_rpn_scores = tf.gather(rpn_scores_nms, positive_indices)
    negative_rpn_scores = tf.gather(rpn_scores_nms, negative_indices)
    positive_corresponding_anchors = tf.gather(corresponding_anchors, positive_indices)
    negative_corresponding_anchors = tf.gather(corresponding_anchors, negative_indices)
    """以上，找到正例和负例对应的rpn得分，和他们对应的锚。"""
    all_indices_bool = tf.ones([config.POST_NMS_ROIS_TRAINING], dtype=bool)  # config.POST_NMS_ROIS_TRAINING（100）个1
    all_indices = tf.where(all_indices_bool)[:, 0]  # 这个是0~99
    c1 = tf.sets.set_difference(all_indices[None, :], positive_indices[None, :])  # 没有被选为正例的那些东西
    not_positive_indices = c1.values  # 所有的非正例，因为非正例多，所以negative_indices里没有选完，这儿是除了正例之外的所有索引。
    not_positive_rois = tf.gather(proposals, not_positive_indices)  # 非正例的提出，就是所有提出去掉正例提出剩下的那些。

    positive_overlaps = tf.gather(overlaps, positive_indices)
    # 上句，执行后shape=(?,?)，这两个?应该分别是positive_count（选定了的正例个数）和V_g-V_c，
    #     然后得到的positive_overlaps矩阵中的一行，应该是某一个正例提出和所有金标准外接矩阵的IOU。
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    # 上句，执行后shape=(?,)，这个?是positive_count（选定了的正例个数）。 --<tf.Tensor 'mask_rcnn_model/ArgMax_1:0' shape=(?,) dtype=int64>
    #     得到结果是（positive_overlaps矩阵中的）每个正例提出，与各个金标准外接矩形（已经去除了拥挤矩形的）的最大重合度。
    #     这就相当于是把每一个RPN提出弄出来的正例，对应到了某个金标准提出（就是外接矩形重合最多的那个）上。
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    # 上句，提取出最大交并比的金标准外接矩形。shape是(?,4)，?对应positive_count（选定了的正例个数），--<tf.Tensor 'mask_rcnn_model/Gather_14:0' shape=(?, 4) dtype=float32>
    #     这个roi_gt_boxes的每一列都表示一个正例对应的金标准提出的外接矩形。
    #     这就相当于是找到了对应的金标准提出后，弄出来这个金标准提出的外接矩形。
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
    # 上句，提取出最大交并比的金标准类别。shape是(?,)，?对应positive_count个正例提出的类别。理解同上。 -- <tf.Tensor 'mask_rcnn_model/Gather_15:0' shape=(?,) dtype=int32>
    """
    以上，把刚刚弄出来的每一个正例，对应到了某个金标准提出上，
        即：通过外接矩形的重合度，选出了每个被选定的正例（一共有positive_count个）的金标准类别和外接矩形。
    """
    # ix = tf.nn.top_k(roi_gt_class_ids, k=tf.to_int32(tf.shape(positive_indices)[0])).indices[::-1]  # <tf.Tensor 'mask_rcnn_model/TopKV2_1:1' shape=(?,) dtype=int32>
    # roi_gt_class_ids = tf.gather(roi_gt_class_ids, ix)  # <tf.Tensor 'mask_rcnn_model/Gather_16:0' shape=(?,) dtype=int32>
    # roi_gt_boxes = tf.gather(roi_gt_boxes, ix)  # <tf.Tensor 'mask_rcnn_model/Gather_17:0' shape=(?, 4) dtype=float32>
    # positive_rois = tf.gather(positive_rois, ix)  # <tf.Tensor 'mask_rcnn_model/Gather_17:0' shape=(?, 4) dtype=float32>
    # roi_gt_box_assignment = tf.gather(roi_gt_box_assignment, ix)  # <tf.Tensor 'mask_rcnn_model/Gather_18:0' shape=(?,) dtype=int64>
    # positive_rpn_scores = tf.gather(positive_rpn_scores, ix)
    # negative_rpn_scores = tf.gather(negative_rpn_scores, ix)
    # positive_corresponding_anchors = tf.gather(positive_corresponding_anchors, ix)
    # negative_corresponding_anchors = tf.gather(negative_corresponding_anchors, ix)
    # """
    # 以上，把对应的金标准的顺序排一下，然后对应的正例提出的顺序也要变一下。
    # 变了之后，这三个正例提出（及其对应的金标准类别/外接矩形/掩膜）的类别应该是1 2 3这样按照类别序号的顺序排下去的。
    # [::-1]是反序的意思。
    # """。。。。。。后来觉得训练的时候这么排序根本没有意义，因为测试的时候仍然不能保证是按顺序排的，所以删掉了。
    roi_gt_sparse = tf.gather(gt_sparse, roi_gt_box_assignment)

    deltas = MRCNN_utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    # shape是(?,4)，和model.py里的那个apply_box_deltas_graph不一样，见函数里的注释。
    deltas /= config.BBOX_STD_DEV  # shape是(?,4)。
    """以上，根据前面弄出来的、每个被选定的正例的RPN提出的外接矩形和金标准外接矩形，计算出来外接矩形修正值deltas。"""

    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # 上句，执行后shape=(?, 56, 56, 1)。那个?就是V_g-V_c，去除了拥挤矩形后的金标准提出个数。
    #     正好解决了前面的疑惑，其实就是他把这个个数放到最后了。。。
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    # 上句，提取出每个被选定的正例的、与之交并比最大的、金标准掩膜。
    #     执行后shape是(?,56,56,1)，此时的?是positive_count（选定了的正例个数），最后的1没用（后面会squeeze掉）。
    #     也就是说，roi_masks是positive_count个56*56的矩阵，每个矩阵就代表一个被选定的正例的金标准掩膜。
    """以上，选出了每个被选定的正例（一共有positive_count个）的金标准掩膜。"""

    boxes = positive_rois  # 正例的、RPN网络提出的、外接矩形。shape=(?,?)，具体见上。
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])  # 这个就是选定了的正例个数（positive_count）吧，就是0~positive_count-1。
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE)
    # 上句，把输入的image（一张图，这儿是变成float类型后的roi_masks）裁剪，保留的区域是boxes里定义的区域
    #     （boxes作为外接矩形，这儿就是用RPN提出的外接矩形作为外接矩形了，相当于是把RPN提出出来的东西给截下来了）。
    #     然后box_ind是这张图里裁剪出来的区域数（这儿就是那个提出数0~positive_count-1），crop_size是把裁剪出来的区域放缩到这个大小。
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)
    # 上句，把masks改成(?,56,56)的。

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    # 上句，正例和负例的、RPN网络提出的、外接矩形。
    # 第一次是<tf.Tensor 'mask_rcnn_model/concat_1:0' shape=(?, ?) dtype=float32>，第一个?应该就是positive_count+negative_count，
    #     表示选定了这么多个正例和负例；第二个?应该是4，表示4个角点。
    # 第二次是<tf.Tensor 'mask_rcnn_model/concat_6:0' shape=(?, ?) dtype=float32>
    all_corresponding_anchors_selected = tf.concat([positive_corresponding_anchors, negative_corresponding_anchors], axis=0)
    all_rpn_scores_selected = tf.concat([positive_rpn_scores, negative_rpn_scores], axis=0)
    proposals_ordered = tf.concat([positive_rois, not_positive_rois], axis=0)  #
    """上句是新加的，这个proposals_ordered应该和输入的proposals一样，只不过是前几行应该和rois及roi_gt_class_ids是对应的。"""
    N = tf.shape(negative_rois)[0]  # shape=()，就是一个数。应该就是negative_count，负例数。
    # 无论何时，N都是<tf.Tensor 'mask_rcnn_model/strided_slice_102:0' shape=() dtype=int32>
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    # 上句，<tf.Tensor 'mask_rcnn_model/Maximum_18:0' shape=() dtype=int32>，不是正例数，而是补零数。想让后面的rois一共有config.TRAIN_ROIS_PER_IMAGE行。
    #  无论何时，P都是<tf.Tensor 'mask_rcnn_model/Maximum_18:0' shape=() dtype=int32>。
    # 如果config.ONE_INSTANCE_EACH_CLASS=True、且是训练好了的话，这个P其实应该是0了。
    #     但是，即使config.ONE_INSTANCE_EACH_CLASS=True，如果刚刚开始训练，没有正例，也就没有负例，那么P应该就是config.TRAIN_ROIS_PER_IMAGE。
    # 【基础】注意这种方法！！要用tf.shape(rois)[0]，而不是rois.shape[0]或者tf.cast(rois.shape[0], dtype=tf.int32)！！！！
    rois = tf.pad(rois, [(0, P), (0, 0)])
    # 上句，<tf.Tensor 'mask_rcnn_model/Pad_3:0' shape=(?, ?) dtype=float32>，第一个?就是config.TRAIN_ROIS_PER_IMAGE（即positive_count+negative_count+P）
    # 第二次是 <tf.Tensor 'mask_rcnn_model/Pad_11:0' shape=(?, ?) dtype=float32>
    P1 = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(positive_rois)[0], 0)
    positive_rois = tf.pad(positive_rois, [(0, P1), (0, 0)])  # 类似，给positive_rois也补零啊（否则run掉的时候，如果这个批次里每张图的正例提出数不一样，run的时候就报错）
    all_corresponding_anchors_selected = tf.pad(all_corresponding_anchors_selected, [(0, P), (0, 0)])
    all_rpn_scores_selected = tf.pad(all_rpn_scores_selected, [(0, P)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])  # shape=(?,)
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])  # shape=(?,?)
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])  # shape=(?,?,?)
    roi_gt_sparse = tf.pad(roi_gt_sparse, [(0, N + P), (0, 0)])
    # 以上几行，重新补零回来。

    """
    【总之】根据RPN弄的区域提出ROI，还有金标准类别、外接矩形、掩膜，先从区域提出ROI中把正例和负例找出来，
        然后随机选出来几个（即所谓下采样），然后，一方面返回这些正例和负例rois，
        一方面在金标准中找到这些正例和负例对应的金标准类别target_class_ids、金标准外接矩形修正值target_ deltas、金标准掩膜target_mask。
    """
    # 下面return,注意DetectionTargetLayer里的names的个数得和现在return回去的个数一样多！
    return rois, all_rpn_scores_selected, all_corresponding_anchors_selected, roi_gt_class_ids, deltas, masks, roi_gt_sparse, proposals_ordered