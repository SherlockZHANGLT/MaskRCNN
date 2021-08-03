#coding=utf-8
import tensorflow as tf
import MRCNN_utils
def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    锚的分类损失
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
               输入的rpn_match：shape是[?,?,1]，前两个?分别为金标准输入中批次里的图像张数，金标准输入中每张图里的锚个数（如65472个锚），
               锚匹配的种类：1是正样本，-1是负样本，0是中性样本。
               在调用的时候，输入的是input_rpn_match，这是在“一、RPN GT”里输入进来的，用作金标准的分类。
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
               输入的rpn_class_logits：RPN网络的分类结果，也就是【第三步】的输出。
               在调用的时候，输入进来的是那个rpn_class_logits，这是第三步的结果，即，从不同尺度的特征图中提取出来的东西，
                   此时还没做细化、裁剪、非最大抑制， 也还没用那个锚点（这些都是第四步做的）。
               这个rpn_class_logits的shape=(?, ?, 2)，前两个?分别表示批次里的图像张数，每张图里的锚个数，
                   然后这个2就是某个锚为负例和正例的概率，相当于是个1热标签吧。
    """
    rpn_match = tf.squeeze(rpn_match, -1)
    # 上句，把rpn_match最后一维去掉。
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # 上句，找到rpn_match中等于1的位置，然后变成int32类型。所以anchor_class里为1的地方，就是rpn_match里为1的地方；
    #     anchor_class里为0的地方，就是rpn_match里为0或-1的地方。
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # 上句，找到非中性样本的索引号，即rpn_match（金标准）里为1或-1的地方。shape应该是(?, 2)。
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)  # shape=(?, 2) dtype=float32
    anchor_class = tf.gather_nd(anchor_class, indices)  # shape=(?,) dtype=int32
    # 上两句，根据索引号，找到非中性样本的RPN分类和金标准分类。
    #     这个时候，anchor_class里为1的地方，就是rpn_match里为1的地方；为0的地方，就是rpn_match里为-1的地方。这个anchor class里应该要么是0要么是1，否则loss可能变nan。
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = rpn_class_logits, labels = anchor_class)
    # 上句，弄交叉熵损失。好像是说，anchor_class是金标准，然后rpn_class_logits是网络预测值。
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.reduce_mean(loss), tf.constant(0.0))  # shape=() dtype=float32
    # 上句，如果小于0的话，就记作0。
    """
    以上，是对比以下两个：1、RPN网络弄出来的锚（rpn_class_logits）是正例还是负例；2、金标准锚（anchor_class）是正例还是负例。
        对比完了就得到锚的分类损失，即，只看是正例还是负例，没管到底是哪一类。
    那个金标准锚，是class MaskRCNN输入中的input_rpn_match，而这正是data_generator函数弄出来的。
    """
    other_outputs = [rpn_match, indices, rpn_class_logits, anchor_class]
    return loss, other_outputs

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    返回RPN外接矩形损失的计算图

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
        输入的target_bbox：金标准外接矩形修正值，shape=(?, ?, 4)，两个?分别为批次中图像张数、一张图中的最大正例数（注意到，
            这个最大正例数和rpn_bbox中的anchors、即一张图中预测出来的锚数，以及后面那个batch_counts、即预测出来的正例数，是不一样的，
            所以需要从这里面选出来batch_counts个）。
        在调用的时候，输入进来的是input_rpn_bbox，这是在“一、RPN GT”里输入进来的，用作金标准外接矩形。
            考虑到这个整个函数返回的是个模型，估计是可以用作模型输入的。详见【基础】Keras里的Model。
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
        输入的rpn_class_logits：和上面的那个函数应该是一样的，RPN网络的分类结果。对于一张图，就是个65472维的向量，全是1、-1、0这仨数。
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        输入的rpn_bbox：应该是RPN网络弄出来的外接矩形修正值。
        在调用的时候，输入进来的是那个rpn_bbox，这是第三步的结果，即，从不同尺度的特征图中提取出来的东西，
            此时还没做细化、裁剪、非最大抑制， 也还没用那个锚点（这些都是第四步做的）。
        对比一下第三步的rpn_graph函数里的rpn_bbox，它的shape=(?, ?, 4)，前两个?分别表示批次里的图像张数，每张图里的锚个数。
            似乎和上面[batch, max positive anchors, (dy, dx, log(dh), log(dw))]描述的一致。
    """
    rpn_match = tf.squeeze(rpn_match, -1)  # 把rpn_match最后一维去掉。同前。
    indices = tf.where(tf.equal(rpn_match, 1))  # shape=(?,2)，已验证。
    # 上句，找到rpn_match为1的，即金标准中正例样本的索引号。因为，只有正例的锚才对外接矩形损失函数有贡献，负例和中性样本的锚都没有贡献。（这个可以理解）
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)  # shape=(?,4)，已验证。
    # 上句，用索引号选出正例锚。注意到，indices是用金标准的类别判定选出来的，而rpn_bbox是预测的外接矩形修正值。
    #   也就是说，是用金标准正例序号，从预测外接矩形中选择的。
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)  # shape=(?,)，已验证。
    # 上句，找出来每一张图的预测结果中有多少个正例锚，?应该是批次中的图张数。
    target_bbox = MRCNN_utils.batch_pack_graph(target_bbox, batch_counts, config.BATCH_SIZE)  # shape=(?,4)，已验证。config.IMAGES_PER_GPU改成了config.BATCH_SIZE
    # 上句，把target_bbox修剪，让他的长度为batch_counts（和rpn_bbox等长）。
    #     暂时理解为：target_bbox里是所有正例锚的金标准外接矩形修正值，
    #     但是呢，rpn_bbox可能并不是把所有的正例都弄出来了，所以就从金标准外接矩形target_bbox中选出“rpn_bbox里弄出来了的”那些正例，去进行后续计算（算损失啥的）。
    #     但事实上有个问题就是，你怎么保证选出来的那batch_counts个就能和rpn_bbox里的那batch_counts个正好一一对应上呢？
    #     →→→因为在第零步的时候，那个外接矩形修正就已经选好了啊。
    #     还记得那个build_rpn_targets函数里，batch_rpn_match的shape是(8, 65472, 1)，而batch_rpn_bbox的shape是(8, 256, 4)，
    #     这是说，那个金标准的RPN外接矩形已经是只考虑正例的了（一张图中的bbox数已经比锚数少了太多了）。
    # 【但这个解释也未必合适，因为既然这样的话，干什么还要用batch_pack_graph去修剪？也许真的是靠训练对应上的？】
    diff = tf.abs(target_bbox - rpn_bbox)  # shape=(?,4)，已验证。
    # 上句，金标准外接矩形和RPN的外接矩形的差值绝对值（只考虑正例的）。shape=(?,4)，已验证。应该是2个角点的4个坐标都算了的。
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")  # shape=(?,4)，已验证。
    # 上句，找到差值小于1的位置。
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)  # shape=(?,4)
    # 这是《155-参考2：Fast RCNN原文2015.pdf》中的3式。应该学到python里实现那个分段函数的方法。
    #     现在loss的shape=(?,4)，下一步求了平均，就得到4个角点坐标的平均损失，再让他大于0。
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.reduce_mean(loss), tf.constant(0.0))  # shape=()，已验证。
    other_outputs = [rpn_match, indices, rpn_bbox, batch_counts, target_bbox]
    return loss, other_outputs###############

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.
    掩膜RCNN的分类头的损失。确实是用第六步（头网络的输出结果）做的。

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero padding to fill in the array.
        输入的target_class_ids：类别ID序号，用了补零。
        在调用的时候，输入进来的是target_class_ids，是第五步得到的金标准类别索引号。shape=(4, ?)。
        就是说，第五步通过随机选择一些正例和负例提出，弄出来了他们的ROIS还有相应的金标准类型、外接矩形、掩膜；
        然后第六步又根据那些ROIS，用网络头去预测了类型、外接矩形、掩膜。然后这里在拿他们对比。
        “【从图片到提出的第六步】”处有详细解释。
        target_class_ids的shape=(4, ?)，?代表每张图中的正负样例总数，问题是这玩意为啥不是150啊？可能是因为这个没有经过第六步，是第五步就拿出来了。
            然后里面第i行第j列数，似乎都是0~6里的数之一，如果是0那就是说第i张图中的第j个样例是负例，否则是正例。
    pred_class_logits: [batch, num_rois, num_classes]
        输入的pred_class_logits：预测的类别ID序号，估计也用了补零。
        在调用的时候，输入进来的是mrcnn_class_logits，第六步的输出结果。似乎输入的shape应该是[?,?,?]，三个?分别表示
            批次中的图像张数、每张图中的样例数、类别数。但是，实际上这儿是(? ,150 ,7)，这个150是fpn_classifier_graph那个函数里弄出来的。
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
        输入的active_class_ids：对于在图像数据集中的类，就是1；否则就是0。
        在调用的时候，输入进来的就是active_class_ids，这是从主函数里输入进来的，shape是[?,?]，对应于input_image_meta最后面的那几个数。
        第一个?是批次中图像张数，第二个是每张图中具有的物体种类数，记得在HaNDataset中，是7个种类，然后是7个1（表示这张图里这7种器官都有）。
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')  # shape=(4, ?)已验证，dtype=int64。
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)  # shape=(? ,150)已验证，dtype=int64。
    # 上句shape=(? ,150)。?是每个批次里的图像张数，150是每张图中的提出数，然后这个矩阵中某个数就是某张图、某个提出的、logits最大的那个类别。
    # 其实不就是把那个类别的1热向量变成了一个数嘛。。
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)  # shape=(? ,150)已验证，dtype=int32。
    # 思路可能是，他输入的active_class_ids是当前批次中、所有图中、所有样例中的可能有的类别，而现在是从中选取了预测logits最大的那几个类别。
    #     输入active_class_ids[0]应该是一张图中的“在这张图中的器官索引号”。在这儿只能看到active_class_ids[0]的shape=(?,)，
    #         但是应该能推测，应该是找了一张图的meta，所以这个东西应该就是7个1。
    #     输出shape=(? ,150)，那个?就是一个批次里的图像张数，150就是每张图中的样例数，
    #     所以pred_active是个?*150的矩阵，其中某个位置的数字，应该就是某张图、某个样例的类别（应该是0~6之间的数吧）。
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids,
                                                          logits=pred_class_logits)  # shape=(4 ,?)，已验证。dtype=float32。
    # 上句，用金标准类别索引号（(4, ?)也就是(4, 150)）和预测的类别ID序号（(? ,150 ,7)也就是(4 ,150 ,7)）计算交叉熵。
    #     可能是因为用的是sparse_softmax_cross_entropy_with_logits函数，所以金标准用的是ids，而预测值用的是logits（1热向量）。
    # 输出的shape是(4 ,?)，?表示每张图中的样例数（应该是150），那么loss现在是个4*150的矩阵，其中每个数都是某一张图、某个样例的分类损失。
    pred_active = tf.cast(pred_active, tf.float32)
    loss = loss * pred_active  # shape=(4 ,150)，已验证。
    # 擦除掉不在此图的被激活类别中的类别预测损失。
    # →→→暂时理解为，那个pred_active里全都是0或者1，且shape为(? ,150)即(4 ,150)；loss的shape是(4 ,?)即(4 ,150)
    #     那么，pred_active和loss乘了之后，相当于是逐个元素相乘，然后，原来loss里对应pred_active为0的地方，损失就给清零了。
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)  # shape=()，已验证。
    # 计算平均损失。分子就是清零后的总损失，分母是pred_active中为1的元素个数之和，即，被激活类别的个数。
    # →所以说到这儿，应该就是这个激活类别的理解了，等输进来active_class_ids的时候看看？？？？
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    掩膜RCNN外接矩形细化的损失

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        输入的target_bbox：目标外接矩形，实际上输入的是target_bbox，第五步得到的金标准类别索引号。shape=(4,?,?)。
        第一个?代表每张图中的正负样例总数，第二个?代表4，即角点的4个坐标。
    target_class_ids: [batch, num_rois]. Integer class IDs.
        输入的target_class_ids：目标类别，实际上输入的是target_class_ids，第五步得到的金标准类别索引号。shape=(4, ?)。
        ?代表每张图中的正负样例总数
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        输入的pred_bbox：预测的外接矩形，实际上输入的是mrcnn_bbox，第六步的输出结果。类似于上面，这个的shape=(?,150,7,4)，
        应该也是因为有那个第六步的FPN，所以知道了那个150和7啊。
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))  # shape变成了(?,)，，已验证。?代表这一批次所有图中的样例总数，然后里面每个数应该是0~6之间的数，表示哪个类别。
    target_bbox = tf.reshape(target_bbox, (-1, 4)) # shape变成了(?,4)，已验证。
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.keras.backend.int_shape(pred_bbox)[2], 4)) # shape变成了(?,7,4)，已验证。
    # 以上3行，是把批次中的图像张数和每张图中的样例数合并掉了。这里更就说明了，第五步中每张图的样例数都是?，而第六步中就都是150了。
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # 上句执行后，shape=(?,)，已验证。实际检测了一下，输入进来的target_class_ids是补了0的张量，上句是把补的0删掉。详见《TensorFlow基础-keras的中间层输出》。
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)  # 选出来每个正例提出的类别，shape=(?,)，已验证。
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    # 上句，把整理提出的索引号和类别拼成list（就是一个一个对起来），再弄成tensor，shape=(?,2)，已验证。
    target_bbox = tf.gather(target_bbox, positive_roi_ix)  # 'mask_rcnn_model/Gather_38:0'，shape=(?,4)，已验证。
    pred_bbox = tf.gather_nd(pred_bbox, indices)  # 'mask_rcnn_model/GatherNd_7:0'，shape=(?,4)，已验证。
    # 以上应该是从目标外接矩形和预测外接矩形中选出来类别ID是正数的那些。gather_nd那句是在预测的外接矩形中，选择了金标准类别对应的外接矩形。
    #     这样就保证了预测的类别和金标准类别是一样的。详见《TensorFlow基础-keras的中间层输出》。
    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0, MRCNN_utils.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), tf.constant(0.0))  # shape=<unknown？
    # 上句，计算外接矩形损失。和rpn_bbox_loss_graph函数里的一样，也smooth了一下。
    #     然后那个K.switch是说，如果target_bbox不是空的（size=0），就用那个smooth_l1_loss，否则就是0，
    #     这是说target_class_ids不能全是0（全是0就相当于是没有提出，而全是补的0了），否则就把损失函数记作是0。
    loss = tf.reduce_mean(loss)
    loss = tf.reshape(loss, [1, 1])  # shape=(1, 1)
    other_outputs = [positive_roi_ix, target_bbox, indices, pred_bbox]
    return loss, other_outputs

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    掩膜头的二值交叉熵掩膜损失

    target_masks: [batch, num_rois, height, width]. A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        输入的target_masks：金标准掩膜，一个0-1张量，可能补了零。
        实际上输入的是target_bbox，target_mask，第五步的结果，shape=(4, ?, ?, ?)。三个?分别是一张图中的样本数、掩膜长、掩膜宽。
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        输入的target_class_ids：目标类别，实际上输入的是target_class_ids，第五步得到的金标准类别索引号。shape=(4, ?)。
        ?代表每张图中的正负样例总数。
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
        输入的pred_masks：预测的掩膜，实际上输入的是mrcnn_mask，shape=(?, 150, 28, 28, 7)。
    """
    target_class_ids = tf.reshape(target_class_ids, (-1,))  # shape从(4, ?)变成了(?,)，已验证。把那4张图中的所有target_class_ids合并了。
    mask_shape = tf.shape(target_masks)  # shape变成了(4,)，已验证。就是4个数，即target_masks的shape里的那个4和三个?。
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    # shape变成了(?, ?, ?)，即这一批次所有图中的提出数、掩膜长、掩膜宽。已验证。
    pred_shape = tf.shape(pred_masks)  # shape=(5,)，已验证。就是5个数，应该是?（批次大小）、150、28、28、7这5个。
    pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # shape变成了(?, ?, ?, ?)，已验证。后面3个?其实是28、28、7，第一个?是这一批次所有图中的提出数
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])  # shape仍然是(?, ?, ?, ?)，，已验证。交换不同维度的数据。
    # 以上就是把批次中的图像张数和每张图中的样例数合并掉了，掩膜的比起上面外接矩形的，要换一下预测掩膜的维度。
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    # 以上3句同前面的函数，先删掉补零，然后找到每个正例提出的分类，最后把正例的索引号和类别对起来。
    y_true = tf.gather(target_masks, positive_ix)  # <tf.Tensor 'mask_rcnn_model/Gather_32:0' shape=(?, ?, ?) dtype=float32>
    y_pred = tf.gather_nd(pred_masks, indices)  # <tf.Tensor 'mask_rcnn_model/GatherNd_8:0' shape=(?, ?, ?) dtype=float32>
    # 以上：1、从目标掩膜中选出来类别ID是正数的那些；2、从预测掩膜中选出类别ID是正数的（正例的预测掩膜），并且根据每个正例的类别，从7个掩膜里选一个。
    # 两个的shape都是(?,?,?)，三个?应该分别是这一批次中总提出数、28、28（掩膜长宽），已验证。
    loss = tf.keras.backend.switch(tf.size(y_true) > 0, tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
    # 以上计算掩膜交叉熵。和上面函数有点像，也是在
    loss = tf.reduce_mean(loss)
    loss = tf.reshape(loss, [1, 1])
    return loss

def lspmr_lossFun(fea: object, adjacentMatrix: object) -> object:

    count = tf.clip_by_value(tf.cast(tf.count_nonzero(adjacentMatrix), tf.float32),1.0,100000000.0)

    D = tf.diag(tf.reduce_sum(adjacentMatrix, 1))
    L = D - adjacentMatrix  ## L is laplation-matriX
    loss = 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(fea), L), fea))
    loss = tf.divide(loss,count)

    return loss

