#coding=utf-8
import tensorflow as tf
import MRCNN_utils
import numpy as np

def get_proposal_from_RPN(rpn_class, rpn_bbox, proposal_count, nms_threshold, anchors, config, name):
    # 输入的rpn_class： <tf.Tensor 'mask_rcnn_model/rpn_class:0' shape=(?, ?, 2) dtype=float32>
    #     run掉后发现，shape是(2(批大小), 65472(一开始的锚数), 2)，那么，rpn_class[0,:,:]就是个65472行2列的矩阵，第0和1列分别是该锚属于负例和正例的概率。
    scores = rpn_class[:, :, 1]  # <tf.Tensor 'mask_rcnn_model/strided_slice_20:0' shape=(?, ?) dtype=float32>
    # run掉后发现，shape是(2(批大小), 65472(一开始的锚数))，2行、65472列，2行是2张图，每一行中的每个数都是该图中某个锚是正例的概率。
    deltas = rpn_bbox  # shape应该是(?, ?, 4)  外接矩形长宽和中心位置的修正值。已验证和KL的一样。 dtype=float32
    deltas = deltas * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])  # shape应该是(?, ?, 4)。已验证和KL的一样。
    pre_nms_limit = min(6000, anchors.shape[0])  # self.anchors.shape[0]应该是65472，现在取min就应该是6000。已验证和KL的一样。
    ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name=name + "top_anchors").indices
    # 输出ix是<tf.Tensor 'mask_rcnn_model/ROItop_anchors:1' shape=(?, 6000) dtype=int32>。
    #     run掉后发现，shape是(2(批大小), 6000)，2行，每一行6000个数，都是scores最大的6000个锚的索引号，而且是按照scores从大到小的顺序排列的。
    scores = MRCNN_utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                               config.IMAGES_PER_GPU)  # 上句，scores的shape应该是(4(批大小), 6000)，已验证和KL的一样。
    # <tf.Tensor 'mask_rcnn_model/packed:0' shape=(2, 6000) dtype=float32>
    #     run掉后发现，shape是(2(批大小), 6000)，2行，每一行6000个数，都是scores最大的6000个锚的分数，而且是按照scores从大到小的顺序排列的。
    # 【注意】上句有一个花絮是，一次验证的时候发现是(1, 6000)，这是因为config.IMAGES_PER_GPU是1 。
    #     那是因为，那次验证把MaskRCNN_1_model里的class InferenceConfig中的IMAGES_PER_GPU设成1了。
    deltas = MRCNN_utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),  # 此时deltas的shape应该是(4, 6000, 4)
                               config.IMAGES_PER_GPU)
    # <tf.Tensor 'mask_rcnn_model/mul:0' shape=(?, ?, 4) dtype=float32>
    anchors = MRCNN_utils.batch_slice(ix, lambda x: tf.gather(anchors, x),  # 此时anchors的shape应该是(4, 6000, 4)
                                config.IMAGES_PER_GPU, names=name + "pre_nms_anchors")
    # <tf.Tensor 'mask_rcnn_model/R:0' shape=(2, 6000, 4) dtype=float32>
    """
    所以说，以上三步，其实做的是同一件事，即，根据那个ix，在输入的65472个
        {类别分数、外接矩形长宽和中心位置的修正值、锚（即外接矩形角点坐标）}这三样东西里，取出ix对应的（也就是最有可能是正例的）那6000个。
    然后建立一个小批次，小批次中有self.config.IMAGES_PER_GPU=4张图，每张图有6000个提出，每个提出的分数、外接矩形修正值、锚就是上面三个变量。
    """
    boxes = MRCNN_utils.batch_slice([anchors, deltas],  # 此时boxes的shape=(4, 6000, 4)。验证无误。
                              lambda x, y: MRCNN_utils.apply_box_deltas_graph(x, y),
                              config.IMAGES_PER_GPU,
                              names=name + "refined_anchors")
    # <tf.Tensor 'mask_rcnn_model/R_1:0' shape=(2, 6000, 4) dtype=float32>
    """上句是先做了那个锚修正（用deltas修正了锚点的坐标，就得到了外接矩形了啊），然后建立小批次。
    注意到，输入的那个rpn_bbox其实是外接矩形修正值，而上句得到的boxes则是外接矩形的实际值！
    此处lambda x, y: apply_box_deltas_graph(x, y)的用法是，x就是anchors、即刚才的那个shape=(4, 6000, 4)的张量，
        y就是deltas、即刚才弄好的那个shape=(4, 6000, 4)的张量。然后，把它们输入到那个apply_box_deltas_graph函数里，
        其中输入的anchors在此函数中就是boxes（想想anchors的shape后面的4，不就是角点坐标嘛），然后输入的deltas在此函数中就是deltas，
        再然后根据deltas修正了boxes并且输出为results，这个results在外面就记作了boxes。
    """
    height, width = config.IMAGE_SHAPE[:2]  # 都是512。验证无误。
    window = np.array([0, 0, height, width]).astype(np.float32)  # array([   0.,    0.,  512.,  512.], dtype=float32)。验证无误。
    boxes = MRCNN_utils.batch_slice(boxes,  # 此时boxes的shape仍然=(4, 6000, 4)。验证无误。
                              lambda x: MRCNN_utils.clip_boxes_graph(x, window),
                              config.IMAGES_PER_GPU,
                              names=name + "refined_anchors_clipped")
    """上句是先裁剪外接矩形，然后建立小批次。
    这一步是在上一步细化的基础上，把那个外接矩形给裁剪了。
        【基础】看那个lambda x: clip_boxes_graph(x, window)，能不能看出来点规律？
        →→→本来，这个函数输入是boxes, window，然后那个boxes是个tf张量，可能不能用，所以用前面的boxes和lambda x，来表示这里的x就是那个boxes变成np.array。
        所以说，这个就是先执行了那个clip函数，把那个boxes给过了一遍clip函数，然后再用batch_slice去弄那个批量。
        详见后面“【基础】lambda函数 和 KL.Lambda”那块。
    """
    # TODO: Filter out small boxes滤掉小的外接矩形。原来程序说，由于可能降低小物体的准确性，所以不干了。我在想是不是可以在这儿滤掉两个重复太多的掩膜。
    normalized_boxes = boxes / np.array([[height, width, height, width]])  # 此时normalized_boxes的shape=(4, 6000, 4)。验证无误。
    normalized_anchors = anchors / np.array([[height, width, height, width]])  # 这个应该不用裁剪，因为锚肯定是在原图内部的。

    def nms(normalized_boxes, normalized_anchors, scores):  # 是在下面那个utils.batch_slice(...)里调用的，非最大化抑制。
        indices = tf.image.non_max_suppression(
            normalized_boxes, scores, proposal_count,
            nms_threshold, name=name + "rpn_non_max_suppression")  # nms_threshold是说，如果两个提出的重合度比这个数大，那么就删掉一个。
        proposals = tf.gather(normalized_boxes, indices)
        scores_nms = tf.gather(scores, indices)
        corresponding_anchors = tf.gather(normalized_anchors, indices)
        # Pad if needed
        padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals, scores_nms, corresponding_anchors
    proposals, scores_nms, corresponding_anchors = MRCNN_utils.batch_slice\
        ([normalized_boxes, normalized_anchors, scores], nms, config.IMAGES_PER_GPU)
    """
    上句是先做非最大抑制，然后建立小批次。
    utils.batch_slice([normalized_boxes, scores], nms,...)是把前面得到的[normalized_boxes, scores]这两个东西输入到nms函数中，
        执行完了那个函数就得到proposals，同时通过batch_slice批量化（一个批次里有4个）
    因为现在proposals的长宽都不确定，所以其shape的后两项都是?。

    【总结】这个class ProposalLayer的call函数，就是根据已有的锚（包括它们的类型分值、外接矩形长宽和中心位置的修正值、角点坐标，即anchors\delta\ bbox），
    通过选出来类型分值最大的6000个，然后细化外接矩形、裁剪、非最大化抑制，得到最终的区域提出。同时，弄成一个批次中有4个元素的那种形式。
    """
    return proposals, scores_nms, corresponding_anchors