#coding=utf-8
import tensorflow as tf
import MRCNN_utils
import MaskRCNN_2_ResNet_and_other_FEN as MaskRCNN_2_ResNet
import MaskRCNN_3_RPN
import MaskRCNN_4_Proposals
import MaskRCNN_5_find_gt
import MaskRCNN_6_heads
import MaskRCNN_7_losses
import MaskRCNN_8_detection
import dict_learning_manners

def MaskMRCNN_model(mode, config, global_step, anchors, input_image_pl, input_image_meta_pl, input_rpn_match_pl,
                    input_rpn_bbox_pl, input_gt_class_ids_pl, input_gt_boxes_pl, input_gt_masks_pl, input_sparse_gt, train_flag):
    assert mode in ['training', 'inference']
    h, w = config.IMAGE_SHAPE[:2]  # [:2] 似乎表示config.IMAGE_SHAPE的前2维。现在h和w都是512，就是图像长宽。
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        """
        对图像的长宽是有限制的，长宽都至少是64的倍数。
        """
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    image_scale = tf.cast(tf.stack([h, w, h, w], axis=0), tf.float32)
    gt_boxes_pl = input_gt_boxes_pl / image_scale
    # ###################################################################################################################
    # #                  第0部分，建立各种输入的占位符。现在在主函数中构建占位符，然后传到这个模型里去。                          #
    # ###################################################################################################################
    # input_image = tf.placeholder(tf.float32, [None, h, w, 3])
    # input_image_meta = tf.placeholder(tf.int32, [None, 1+3+4+class_numbers])
    # """上面两句，输入图像和meta。不过，run掉后发现，图像里面好多数是-多少多少诶，一开始没有归一化到0~255吗？？？"""
    # input_rpn_match = tf.placeholder(tf.int32, [None, None, 1])
    # """
    # 上句，input_rpn_match的shape是(?,?,1)。第一个?是批次中的图张数，第二个?是一张图中的锚数（65472），那个1相当于没有。
    #     所以就相当于一个二维矩阵，其中每个数都是金标准的、某个锚是正例还是负例还是中性例的判断。
    # """
    # input_rpn_bbox = tf.placeholder(tf.float32, [None, None, 4])
    # """
    # 上句，input_rpn_bbox的shape是(?,?,4)。第一个?是批次中的图张数，第二个?是训练用的的锚数（256）。
    #     注意，这个N和上面的input_rpn_match中的N是不一样的，这儿是训练用的锚数，上面是总的锚数。
    # 另外，这个input_rpn_bbox不是外接矩形，而是外接矩形修正值。
    # """
    # input_gt_class_ids = tf.placeholder(tf.int32, [None, None], name="input_gt_class_ids")
    # """
    # 上面一段，金标准类别的占位符。input_gt_class_ids的shape是(?,?)。
    # 第一个?是每个批次中的图像张数，第二个?是每张图中的最大金标准提出数（100）。
    # """
    # input_gt_boxes = tf.placeholder(tf.float32, [None, None, 4], name="input_gt_boxes")
    # h, w = tf.shape(input_image)[1], tf.shape(input_image)[2]  # 输入图像的高宽。
    # image_scale = tf.cast(tf.stack([h, w, h, w], axis=0), tf.float32)
    # gt_boxes = input_gt_boxes/image_scale
    # """
    # 上面一段，读出来金标准外接矩形，并且归一化到0~1之间。输出的shape是(?,?,4)，
    #     其中第一个?是批次中的图张数，第二个?是每张图中的最大金标准提出数，同input_gt_class_ids。
    # """
    # if config.USE_MINI_MASK:
    #     input_gt_masks = tf.placeholder(tf.bool, [None, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None], name="input_gt_masks")
    # else:
    #     input_gt_masks = tf.placeholder(tf.bool, [None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks")
    #     # 如果不用小掩膜，input_gt_masks输出的shape是(?, 512, 512, ?)
    # """上面一段弄出来掩膜。其中第一个?是批次中的图张数，第二个?是每张图中的最大金标准提出数（别理解为是通道数）。"""
    ###################################################################################################################
    #                             以上，第0部分结束。 以下第1部分，构建RPN特征图和MRCNN特征图                               #
    ###################################################################################################################
    if config.FEN_network == "Resnet101":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.resnet_graph\
            (input_image_pl, "resnet101", train_flag, stage5=True)  # 残差网络
    elif config.FEN_network == "Resnet50":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.resnet_graph\
            (input_image_pl, "resnet50", train_flag, stage5=True)  # 残差网络
    elif config.FEN_network == "VGG19":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.VGG19(input_image_pl, train_flag=train_flag)  # 用VGG19作为FEN的。
    elif config.FEN_network == "VGG19_FCN":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.VGG19_FCN(input_image_pl, train_flag=train_flag)  # 用VGG19作为FEN的。
    elif config.FEN_network == "VGG19_FCN_simp":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.VGG19_FCN_simp(input_image_pl, train_flag=train_flag)  # 用VGG19作为FEN的。
    elif config.FEN_network == "Googlenet":
        [C1, C2, C3, C4, C5] = MaskRCNN_2_ResNet.GoogleNet_FCN(config, input_image_pl)  # 用Googlenet作为FEN的。
    else:  # 这个else就是Densenet的情况。
        [C1, C2, C3, C4, C5], _ = MaskRCNN_2_ResNet.densenet(input_image_pl, train_flag=train_flag)  # DenseNet的。
    print (config.FEN_network, '执行完毕，共享卷积层已经建立。')
    FEN_results = [C1, C2, C3, C4, C5]
    """
    上句，建立共享卷积层（好像就是174文章中的 backbone architecture、骨架结构，即那个ResNet-101或50-C4或5）
    141论文中表1说得很明白，那个resnet_graph完全是按照这个做的。所以说这里似乎是对输入图像充分地提取了特征。。。
    
    这儿有个网址说了共享卷积层的事儿：https://zhuanlan.zhihu.com/p/35854548似乎不错，还没看。
    """
    # batch_size_test
    P6, P5, P4, P3, P2 = MaskRCNN_2_ResNet.top_down_layers(C5, C4, C3, C2)
    print ('从上向下的层执行完毕。')
    mrcnn_feature_maps = [P2, P3, P4, P5]
    ###################################################################################################################
    #                             以上，第1部分结束。 以下第2+3部分，用锚点和特征图实现区域提出网络。                         #
    ###################################################################################################################
    layer_outputs = []
    with tf.variable_scope("image_filters") as scope:
        p = P2
        # 上句，RPN_ANCHOR_SCALES和BACKBONE_STRIDES里有5个数的时候（BACKBONE_STRIDES = [4, 8, 16, 32, 64]；
        #     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)）是用P2；
        # 然后如果只有中间三个数（BACKBONE_STRIDES = [8, 16, 32]；RPN_ANCHOR_SCALES = (16, 32, 64)）是用P3，
        #     相当于不用P6和P2这两个尺度的特征图。
        rpns_this_level = MaskRCNN_3_RPN.build_rpn_model_with_reuse(p, config.RPN_ANCHOR_STRIDE,
                                                                    len(config.RPN_ANCHOR_RATIOS), 256)  # 改成共享RPN的那个。。
        layer_outputs.append(rpns_this_level)
        scope.reuse_variables()  # 和那个21(reuse_kernels_and_biases).py中的一样。。
        for p in [P3, P4, P5, P6]:
        # 那个for循环，RPN_ANCHOR_SCALES和BACKBONE_STRIDES里有5个数的时候（见上）是用[P3, P4, P5, P6]；
        # 如果只有中间三个数就是[P4, P5]。
            rpns_this_level = MaskRCNN_3_RPN.build_rpn_model_with_reuse(p, config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), 256)  # 改成共享RPN的那个。。
            layer_outputs.append(rpns_this_level)
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [tf.concat((list(o)), axis=1, name=n)
               for o, n in zip(outputs, output_names)]
    """
    上句，是把上一步三堆里的东西给各自合并了，这样下一句就可以直接取出来了。输出是：
    [<tf.Tensor 'mask_rcnn_model/rpn_class_logits:0' shape=(?, ?, 2) dtype=float32>,
      <tf.Tensor 'mask_rcnn_model/rpn_class:0' shape=(?, ?, 2) dtype=float32>,
      <tf.Tensor 'mask_rcnn_model/rpn_bbox:0' shape=(?, ?, 4) dtype=float32>]
    和原程序执行结果对比了一下，发现除了那些name不一样外，其他的都是一样的。
    """
    rpn_class_logits, rpn_class, rpn_bbox = outputs
    ###################################################################################################################
    #                             以上，第3部分结束。 以下第4部分，从锚生成区域提出（选正例锚、细化什么的）。                  #
    ###################################################################################################################
    proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois, rpn_scores_nms, corresponding_anchors = MaskRCNN_4_Proposals.get_proposal_from_RPN\
        (rpn_class, rpn_bbox, proposal_count, config.RPN_NMS_THRESHOLD, anchors, config, name="ROI")
    """
    rpn_rois:Tensor("mask_rcnn_model/packed_2:0", shape=(2, ?, 4), dtype=float32)
    """

    """
    上句，从锚生成区域提出【从图片到提出的第四步】
        根据rpn_class选取若干个最有可能为正例的锚，然后用相应的修正值rpn_bbox去修正、裁剪、非最大抑制，就得到区域提出ROI。
    输出rpn_rois的shape应该是(4,?,?)（已验证无误），那个4是因为IMAGES_PER_GPU = 4（在本py文件最下面，这个4是因为batch_slice函数抛弃掉了原来的批次大小，而重新用IMAGES_PER_GPU分了批次），
        后面两个?应该分别是N和4，分别是每张图中补零后的提出的数量、还有那4个角点坐标。
    """
    ###################################################################################################################
    #         以上，第4部分结束。 以下第5+6+7部分，在训练模式下，从区域提出对应到金标准，然后类别+外接矩形+掩膜回归，             #
    #         即：找到正例锚的金标准类别、外接矩形修正值、掩膜，并且和正例锚的预测类别、外接矩形修正值、掩膜对比。                 #
    ###################################################################################################################
    if mode == "training":
        _, _, _, active_class_ids = MRCNN_utils.parse_image_meta_graph(input_image_meta_pl)  # 输出应该是7个1啥的。。
        rois, rpn_scores_nms_selected, corresponding_anchors_selected, target_class_ids, target_bbox, target_mask, target_sparse, proposals_ordered = \
            MaskRCNN_5_find_gt.DetectionTargetLayer \
            (rpn_rois, rpn_scores_nms, corresponding_anchors, input_gt_class_ids_pl, gt_boxes_pl, input_gt_masks_pl, input_sparse_gt, config)  #
        """
        Tensor("mask_rcnn_model/target_class_ids:0", shape=(2, ?), dtype=int32)
        Tensor("mask_rcnn_model/target_bbox:0", shape=(2, ?, 4), dtype=float32)
        Tensor("mask_rcnn_model/LISTA_dict_learning/transpose_2:0", shape=(?, 2048), dtype=float32)
        Tensor("mask_rcnn_model/target_sparse:0", shape=(2, ?, 2048), dtype=float32)
        """

        """
        以上，输出为：
        rois: <tf.Tensor 'mask_rcnn_model/rois:0' shape=(4/2, ?, ?) dtype=float32>，
            run掉后发现shape是(batch_size, TRAIN_ROIS_PER_IMAGE, 4)。
        target_class_ids: <tf.Tensor 'mask_rcnn_model/target_class_ids:0' shape=(4/2, ?) dtype=int32>，run掉后发现shape是(batch_size, TRAIN_ROIS_PER_IMAGE)
        target_bbox: <tf.Tensor 'mask_rcnn_model/target_bbox:0' shape=(4/2, ?, ?) dtype=float32>
        target_mask: <tf.Tensor 'mask_rcnn_model/target_mask:0' shape=(4/2, ?, ?, ?) dtype=float32>，run掉后发现shape是(batch_size, TRAIN_ROIS_PER_IMAGE, 28, 28)
        rpn_scores_nms_selected：<tf.Tensor 'mask_rcnn_model/rpn_scores_selected:0' shape=(2, ?) dtype=float32>
        corresponding_anchors_selected：<tf.Tensor 'mask_rcnn_model/corresponding_anchors_selected:0' shape=(2, ?, ?) dtype=float32>
        对比了一下原来，至少形状没错。。
        然后输入的几个东西：
        rpn_rois: <tf.Tensor 'mask_rcnn_model/packed_2:0' shape=(2, ?, ?) dtype=float32>，run掉后发现shape是(4/2, POST_NMS_ROIS_TRAINING, 4)
        input_gt_class_ids_pl：<tf.Tensor 'input_gt_class_ids:0' shape=(4/2, ?) dtype=int32>，run掉后发现shape是(4/2, 100)
        gt_boxes_pl：<tf.Tensor 'mask_rcnn_model/truediv:0' shape=(4/2, ?, 4) dtype=float32>
        input_gt_masks_pl：<tf.Tensor 'input_gt_masks:0' shape=(4/2, 56, 56, ?) dtype=bool>，run掉后发现shape是(2, 56, 56, 100)。
        """
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, sparse, shared, D = MaskRCNN_6_heads.fpn_classifier_graph \
            (config, rois, mrcnn_feature_maps, config.TRAIN_ROIS_PER_IMAGE, train_flag)
        """
        Tensor("mask_rcnn_model/mrcnn_class_logits_ensemble/mrcnn_class_logits:0", shape=(2, 60, 10), dtype=float32)
        Tensor("mask_rcnn_model/mrcnn_bbox_logits_ensemble/mrcnn_bbox:0", shape=(2, 60, 10, 4), dtype=float32)
        Tensor("mask_rcnn_model/LISTA_dict_learning/transpose_2:0", shape=(?, 2048), dtype=float32)
        """
        """
        上句输入：rois的shape=(4, ?, ?)，mrcnn_feature_maps就是P5~P2，shape分别为shape=(8, 128, 128, 256)~(8, 16, 16, 256)，
            rois这个东西，run掉之后是什么样的？ → 是(batch_size, TRAIN_ROIS_PER_IMAGE, 4)
        输出：
        mrcnn_class_logits的名字是<tf.Tensor 'mask_rcnn_model/mrcnn_class_logits:0' shape=(?, config.TRAIN_ROIS_PER_IMAGE, NUM_CLASSES) dtype=float32>
            训练的时候run掉后，就是shape=(?, 60, 10)
        mrcnn_class的名字是<tf.Tensor 'mask_rcnn_model/mrcnn_class:0' shape=(?, config.TRAIN_ROIS_PER_IMAGE, NUM_CLASSES) dtype=float32>
        mrcnn_bbox的名字是<tf.Tensor 'mask_rcnn_model/mrcnn_bbox:0' shape=(?, config.TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, 4) dtype=float32>
        """
        with tf.variable_scope("mask") as scope:
            mrcnn_mask = MaskRCNN_6_heads.build_fpn_mask_graph_with_reuse\
                (rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.MASK_POOL_SIZE, config.NUM_CLASSES, config.TRAIN_ROIS_PER_IMAGE, train_flag)
            """mrcnn_mask的执行结果是：
             <tf.Tensor 'mask_rcnn_model/mask/mrcnn_mask:0' shape=(?, TRAIN_ROIS_PER_IMAGE, 28, 28, NUM_CLASSES) dtype=float32>
            """
            scope.reuse_variables()  # 这儿也加一个试试？

        target_class = tf.reshape(target_class_ids, shape=[1, config.BATCH_SIZE * config.TRAIN_ROIS_PER_IMAGE], name='target_class')

        target_class_tile = tf.tile(target_class, [config.BATCH_SIZE * config.TRAIN_ROIS_PER_IMAGE, 1], name='target_class_tile')   #'mask_rcnn_model/target_class_tile:0'

        adjace_matrix = tf.equal(target_class_tile, tf.transpose(target_class_tile), name='adjace_matrix')      #'mask_rcnn_model/adjace_matrix:0'

        adjace_matrix = tf.cast(adjace_matrix, tf.float32)

        adjace_matrix_diag = tf.matrix_diag(tf.diag_part(adjace_matrix))

        adjace_matrix = tf.subtract(adjace_matrix, adjace_matrix_diag)      #Tensor("mask_rcnn_model/Sub_70:0", shape=(120, 120), dtype=float32)
        ################################################################################################################
        ################################################################################################################
        #                 以上，训练模式第5~7部分结束。 以下在if mode == "training"的前提下，计算五种损失。                   #
        ################################################################################################################
        rpn_class_loss, others_rpnclass = MaskRCNN_7_losses.rpn_class_loss_graph(input_rpn_match_pl, rpn_class_logits)
        rpn_bbox_loss, others_rpnbox = MaskRCNN_7_losses.rpn_bbox_loss_graph(config, input_rpn_bbox_pl, input_rpn_match_pl, rpn_bbox)
        class_loss = MaskRCNN_7_losses.mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids)
        bbox_loss, others_mrcnnbox = MaskRCNN_7_losses.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
        mask_loss = MaskRCNN_7_losses.mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)
        lspmr_loss = config.lspmr_weight * MaskRCNN_7_losses.lspmr_lossFun(shared, adjace_matrix)
        if config.USE_LISTA:
            sparsity_loss = MRCNN_utils.smooth_l1_loss(y_true=target_sparse, y_pred=tf.reshape(sparse, [config.BATCH_SIZE, config.TRAIN_ROIS_PER_IMAGE, config.IMAGE_MIN_DIM*config.projection_num]))
            sparsity_loss = tf.reduce_mean(sparsity_loss) * config.weights_sparsity
            sparse_loss = sparsity_loss
            class_loss = class_loss * config.projection_num  # 因为是多次监督，所以乘以投影数。见MaskRCNN_6_heads里的相关叙述。
            bbox_loss = bbox_loss * config.projection_num  # 同上。
        else:
            sparse_loss = tf.constant(0.0)
            sparsity_loss = tf.constant(0.0)



        loss = rpn_class_loss + rpn_bbox_loss + class_loss + bbox_loss + mask_loss + sparse_loss +lspmr_loss
        ################################################################################################################
        #                          以上，计算损失和总损失结束。 设计train_op的事情单独开一个函数来搞了。                      #
        ################################################################################################################
        detections = MaskRCNN_8_detection.DetectionLayer(rois, mrcnn_class, mrcnn_bbox, input_image_meta_pl, config)
        # 上句，和测试的时候差不多，只不过第一个输入用rois而不是rpn_rois，毕竟是细化过的啊。
        # 如果每个金标准只取和它IoU最大的提出，那么，detections中非零行数应该小于等于金标准数。

        """【复用变量】
        在主函数里写共用变量reuse=tf.AUTO_REUSE，是不行的。因为主函数里写，是说比如说主函数里调用了两次MaskMRCNN_model函数，这两次调用的变量一样，
            而现在是希望，在这个MaskMRCNN_model里调用两次build_fpn_mask_graph_with_reuse函数，让这两次调用的变量一样。
        其实忽然又想，是不是也没必要再执行一遍了，就直接用那个mrcnn_mask（而不是mrcnn_mask1）是不是也可以？
        """
        detection_envelop_all = []
        mrcnn_boxes_and_scores = []
        mrcnn_boxes_and_scores_sorted =[]
        for i in range(config.BATCH_SIZE):
            # 原来（备份1、还有滑脱的那些备份里）用过unmold_detections_for_GAN函数去得到boxes_tf之类的，然后去求所有检测物体的外包络。
            # 现在觉得，既然每个金标准只取和它IoU最大的提出，那么，训练的时候、detections中就不会有一个金标准对应多个检测结果的情况了
            #     （刚训练的时候，可能一个金标准对应0个检测结果，但这似乎也没关系）。所以就把训练时候的删掉了，精简程序。测试的时候保留了。
            # 试了一下，用split2的，MRCNN训练了12000次，然后滑脱搞了300次吧，滑脱就能把每个批次的2张图都弄对了。
            detection_trimmed, _ = MRCNN_utils.trim_zeros_graph(detections[i,:,:], name='detection_trimmed')
            # 上句，训练完了shape应该是(3, 4)，但，刚训练的时候可能是0。不过好像也没啥关系，反正刚训练的时候也不会用来搞别的东西。
            y1 = detection_trimmed[:, 0]
            y2 = detection_trimmed[:, 2]
            x1 = detection_trimmed[:, 1]
            x2 = detection_trimmed[:, 3]
            y1 = tf.clip_by_value(y1, 0, h - 1)  # 让所有的角点都在0~511之间。
            y2 = tf.clip_by_value(y2, 0, h - 1)
            x1 = tf.clip_by_value(x1, 0, w - 1)
            x2 = tf.clip_by_value(x2, 0, w - 1)
            detection_envelop_y1 = tf.reduce_min(y1)  # 三块椎骨的左上角点y坐标中最靠上的那个。
            detection_envelop_y2 = tf.reduce_max(y2)  # 三块椎骨的右下角点y坐标中最靠下的那个。
            detection_envelop_x1 = tf.reduce_min(x1)  # 三块椎骨的左上角点x坐标中最靠左的那个。
            detection_envelop_x2 = tf.reduce_max(x2)  # 三块椎骨的右下角点x坐标中最靠右的那个。
            detection_envelop = [detection_envelop_y1, detection_envelop_x1, detection_envelop_y2, detection_envelop_x2]
            detection_envelop_all.append(detection_envelop)
            # 下面把提出修正成最终预测结果，并弄成batch。
            class_ids_this = tf.argmax(mrcnn_class[i, :, :], axis=1, output_type=tf.int32)  # 每个提出的类别预测值
            indices_this = tf.stack([tf.range(mrcnn_class[i, :, :].shape[0]), class_ids_this], axis=1)  # 得1张图1张图地弄。。。
            class_scores_this = tf.gather_nd(mrcnn_class[i, :, :], indices_this)
            mrcnn_bbox_specific_this = tf.gather_nd(mrcnn_bbox[i, :, :, :], indices_this)
            # 上句，MRCNN的外接矩形修正值是对每一类分别预测的，即一个预测对应num_class个修正值，现在要根据MRCNN预测的类别选出来一个修正值。
            mrcnn_boxes_this = MRCNN_utils.apply_box_deltas_graph(rois[i, :, :], mrcnn_bbox_specific_this * config.BBOX_STD_DEV)
            mrcnn_boxes_and_scores_this = tf.concat([mrcnn_boxes_this,
                                                 tf.expand_dims(tf.to_float(class_ids_this), axis=1),
                                                 tf.expand_dims(class_scores_this, axis=1)],
                                                axis=1)  # tf.concat是个烦人的东西。。
            mrcnn_boxes_and_scores.append(mrcnn_boxes_and_scores_this)
            gt_class_positive = tf.where(class_ids_this > 0)[:, 0]  # 用金标准选出来正例
            ix = tf.nn.top_k(target_class_ids[i, :], k=tf.shape(gt_class_positive)[0]).indices[::-1]  # [::-1]是反向，原来是从大到小，让他反过来
            mrcnn_boxes_and_scores_sorted_this = tf.gather(mrcnn_boxes_and_scores_this, ix)
            mrcnn_boxes_and_scores_sorted.append(mrcnn_boxes_and_scores_sorted_this)

        return rpn_class, rpn_bbox, mrcnn_class, mrcnn_bbox, mrcnn_mask, lspmr_loss, rpn_class_loss, rpn_bbox_loss, \
               class_loss, bbox_loss, mask_loss, sparse_loss, sparsity_loss, loss, mrcnn_boxes_and_scores, mrcnn_boxes_and_scores_sorted


    else:
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, sparse, shared, D = MaskRCNN_6_heads.fpn_classifier_graph \
            (config, rpn_rois, mrcnn_feature_maps, proposal_count, train_flag)  #
        """
        上句各种shape：
            mrcnn_class_logits--shape=(?, config.POST_NMS_ROIS_INFERENCE, config.NUM_CLASSES)、
            mrcnn_class--shape=(?, config.POST_NMS_ROIS_INFERENCE, config.NUM_CLASSES)、
            mrcnn_bbox--shape=(?, config.POST_NMS_ROIS_INFERENCE, config.NUM_CLASSES, 4),
        【注意】那个proposal_count，曾经错为config.DETECTION_MAX_INSTANCES，结果run的时候报错。
            这是因为，倒数第二个参数，应该和第一个参数的shape[1]相等，即，输入的东西有多少行，亦即一张图里有多少个提出。
            所以，如果输入是rpn_rois，那就应该是proposal_count； 如果输入是detection_boxes（但这种情况是不可能的，因为是
                先弄的mrcnn_class/bbox才有的detection_boxes）， 那倒数第二个参数才应该是config.DETECTION_MAX_INSTANCES。
        """
        detections = MaskRCNN_8_detection.DetectionLayer(rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta_pl, config)  # shape=(2, 100, 6)。
        """注意，mrcnn_bbox是外接矩形修正值，而下面的detection_boxes才是外接矩形。
        仔细看了一下DetectionLayer函数，就是把输入的rpn_rois和外接矩形修正值变成了外接矩形，然后做了非最大化抑制等修饰工作，从输入的1000个提出变成了100个提出。
        """
        detection_boxes = detections[..., :4]  # shape=(2, 100, 4)
        with tf.variable_scope("mask") as scope:
            mrcnn_mask = MaskRCNN_6_heads.build_fpn_mask_graph_with_reuse(detection_boxes, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE, config.NUM_CLASSES, config.DETECTION_MAX_INSTANCES, train_flag)
            # 上面的mrcnn_mask，shape=(?, config.DETECTION_MAX_INSTANCES, 28, 28, config.NUM_CLASSES)
            scope.reuse_variables()
        mrcnn_boxes_and_scores = []
        for i in range(config.BATCH_SIZE):
            # 下面把提出修正成最终预测结果，并弄成batch。
            class_ids_this = tf.argmax(mrcnn_class[i, :, :], axis=1, output_type=tf.int32)  # 每个提出的类别预测值
            indices_this = tf.stack([tf.range(mrcnn_class[i, :, :].shape[0]), class_ids_this], axis=1)  # 得1张图1张图地弄。。。
            class_scores_this = tf.gather_nd(mrcnn_class[i, :, :], indices_this)
            mrcnn_bbox_specific_this = tf.gather_nd(mrcnn_bbox[i, :, :, :], indices_this)
            # 上句，MRCNN的外接矩形修正值是对每一类分别预测的，即一个预测对应num_class个修正值，现在要根据MRCNN预测的类别选出来一个修正值。
            mrcnn_boxes_this = MRCNN_utils.apply_box_deltas_graph(rpn_rois[i, :, :], mrcnn_bbox_specific_this * config.BBOX_STD_DEV)
            mrcnn_boxes_and_scores_this = tf.concat([mrcnn_boxes_this,
                                                 tf.expand_dims(tf.to_float(class_ids_this), axis=1),
                                                 tf.expand_dims(class_scores_this, axis=1)],
                                                axis=1)  # tf.concat是个烦人的东西。。
            mrcnn_boxes_and_scores.append(mrcnn_boxes_and_scores_this)
        mrcnn_boxes_and_scores = tf.stack(mrcnn_boxes_and_scores, axis=0)
        mrcnn_boxes = mrcnn_boxes_and_scores[..., :4]  # shape=(2, 100, 4)
        with tf.variable_scope("mask") as scope:
            scope.reuse_variables()
            mrcnn_mask_mp = MaskRCNN_6_heads.build_fpn_mask_graph_with_reuse(mrcnn_boxes, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE, config.NUM_CLASSES, config.DETECTION_MAX_INSTANCES, train_flag)
            # 上面的mrcnn_mask，shape=(?, config.DETECTION_MAX_INSTANCES, 28, 28, config.NUM_CLASSES)
        return mrcnn_feature_maps, detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, mrcnn_mask_mp, \
               rpn_rois, rpn_class, rpn_bbox, rpn_scores_nms, mrcnn_boxes_and_scores

def train_MaskMRCNN_model(config, vars, num_samples_per_epoch, loss, global_step):  # 训练上面的模型，得到train_op。原来是把二者合一的，感觉不太好就拆开了。
    """【基础】全局步数
    global_step必须放在这里面，才能起到作用。。
    """
    # 在config.py里的class Config加了一些训练参数，如num_epochs_per_decay等，都是小写字母的。
    decay_steps = int(num_samples_per_epoch / config.BATCH_SIZE * config.num_epochs_per_decay)
    # 上句，通过1个时代中的样本数，还有1个批次中的样本数，就能算出1个时代有多少个批次，然后再除以多少时代衰减1次学习率，就能知道过多少步衰减一次学习率。
    if config.learning_rate_decay_type == 'exponential':
        configured_learning_rate = tf.train.exponential_decay(config.LEARNING_RATE, global_step, decay_steps,
                                                              config.learning_rate_decay_factor, staircase=True,
                                                              name='exponential_decay_learning_rate')
    elif config.learning_rate_decay_type == 'fixed':
        configured_learning_rate = tf.constant(config.LEARNING_RATE, name='fixed_learning_rate')
    elif config.learning_rate_decay_type == 'polynomial':
        configured_learning_rate = tf.train.polynomial_decay(config.LEARNING_RATE, global_step, decay_steps,
                                                             config.end_learning_rate, power=1.0, cycle=False,
                                                             name='polynomial_decay_learning_rate')
    else:
        raise ValueError('无法识别这样的学习率衰减类别(learning_rate_decay_type) [%s]',
                         config.learning_rate_decay_type)

    optimizer = tf.train.MomentumOptimizer(configured_learning_rate, momentum=config.LEARNING_MOMENTUM, name='Momentum')
    grads = optimizer.compute_gradients(loss, var_list=vars)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    """
    【基础】optimizer.minimize和optimizer.apply_gradients
    https://blog.csdn.net/lenbow/article/details/52218551
    首先用tf.train.MomentumOptimizer（或者别的优化方法，如tf.train.GradientDescentOptimizer）函数定义一个optimizer，然后：
        1、可以直接用optimizer.minimize去优化，该函数是简单的合并了compute_gradients()与apply_gradients()函数；
            这个玩意返回的就是train_op。同时，如果输入的global_step非None，该操作还会为global_step做自增操作。
            也就是说，全局步数global_steps在外面主函数里就是个自己定义的普通变量（无论是用global_step = tf.Variable(0, trainable=False)初始化为0
            还是global_step = checkpoint_path_tf.split('/')[-1].split('-')[-1]初始化为以前训练了的全局步数，它都不会自增的），
            只有把它放到这儿之后，才能实现自增操作。
        2、可以先用optimizer.compute_gradients计算梯度，然后对梯度进行操作（我们用的是裁剪，即那个for循环+tf.clip_by_norm）
            再用optimizer.apply_gradients实现最优化，这3步的作用和上面1中的1步的作用是一样的。
            同时，apply_gradients也可以实现全局步数自增操作。
    """
    return train_op, configured_learning_rate