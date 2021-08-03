# coding=utf-8
import tensorflow as tf
import numpy as np
import configure
import make_dataset_from_json
import MaskRCNN_0_get_inputs
import MaskRCNN_1_model
import MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis
import MRCNN_utils
import unmold_MRCNN_outputs_for_visualization
import visualize
import splitting
import pathlib
from tensorflow.python.training import checkpoint_utils as cp
import json
import scipy.io as sio
from keras.utils import to_categorical
from scipy.signal import savgol_filter
import message_passing_manners
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


###################################################################################################################
#                         创建占位符，准备在主函数中把它们输入给MaskRCNN_1_model里的各个变量。                          #
#                         后续滑脱分级网络的占位符创建函数，也写在这儿了。                                              #
###################################################################################################################
def get_placeholder(config, class_numbers, num_rois):
    """MRCNN+MP的占位符。
    """
    h, w = config.IMAGE_SHAPE[:2]  # [:2] 似乎表示config.IMAGE_SHAPE的前2维。现在h和w都是512，就是图像长宽。
    input_image = tf.placeholder(tf.float32, [config.BATCH_SIZE, h, w, 3])  # None能否改成config.BATCH_SIZE？？？
    input_image_meta = tf.placeholder(tf.int32, [config.BATCH_SIZE,
                                                 1 + 3 + 4 + class_numbers])  # None能否改成config.BATCH_SIZE？似乎可以，下同。
    """上面两句，输入图像和meta。不过，run掉后发现，图像里面好多数是-多少多少诶，一开始没有归一化到0~255吗？？？"""
    input_rpn_match = tf.placeholder(tf.int32, [config.BATCH_SIZE, None, 1])
    """
    上句，input_rpn_match的shape是(?,?,1)。第一个?是批次中的图张数（改了之后，第一个?就是8了），第二个?是一张图中的锚数（65472），那个1相当于没有。
        所以就相当于一个二维矩阵，其中每个数都是金标准的、某个锚是正例还是负例还是中性例的判断。
    """
    input_rpn_bbox = tf.placeholder(tf.float32, [config.BATCH_SIZE, None,
                                                 4])  # [config.BATCH_SIZE, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4]
    """
    上句，input_rpn_bbox的shape是(?,?,4)。第一个?是批次中的图张数（改了之后，第一个?就是8了），第二个?是训练用的的锚数（256）。
        注意，这个N和上面的input_rpn_match中的N是不一样的，这儿是训练用的锚数，上面是总的锚数。
    另外，这个input_rpn_bbox不是外接矩形，而是外接矩形修正值。
    """
    input_gt_class_ids = tf.placeholder(tf.int32, [config.BATCH_SIZE, None],
                                        name="input_gt_class_ids")  # [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES]
    """
    上面一段，金标准类别的占位符。input_gt_class_ids的shape是(?,?)。
    第一个?是每个批次中的图像张数（改了之后，第一个?就是8了），第二个?是每张图中的最大金标准提出数（100）。
    """
    input_gt_boxes = tf.placeholder(tf.float32, [config.BATCH_SIZE, None, 4],
                                    name="input_gt_boxes")  # [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 4]
    """
    上面一段，读出来金标准外接矩形，并且归一化到0~1之间。输出的shape是(?,?,4)，
        其中第一个?是批次中的图张数（改了之后，第一个?就是8了），第二个?是每张图中的最大金标准提出数，同input_gt_class_ids。
    原来是在这儿做的归一化，发现不对，就在MaskRCNN_1_model.py的MaskMRCNN_model函数里做了。。
    """
    if config.USE_MINI_MASK:
        input_gt_masks = tf.placeholder(tf.bool,
                                        [config.BATCH_SIZE, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                                        name="input_gt_masks")
    else:
        input_gt_masks = tf.placeholder(tf.bool,
                                        [config.BATCH_SIZE, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                                        name="input_gt_masks")  # [config.BATCH_SIZE, 512, 512, config.DETECTION_MAX_INSTANCES]
        # 如果不用小掩膜，input_gt_masks输出的shape是(?, 512, 512, ?)
    """上面一段弄出来掩膜。其中第一个?是批次中的图张数（改了之后，第一个?就是8了），第二个?是每张图中的最大金标准提出数（别理解为是通道数）。"""
    input_gt_1_hot = tf.placeholder(tf.float32, [config.BATCH_SIZE, config.NUM_CLASSES - 1, config.NUM_CLASSES],
                                    name="input_gt_class_ids")
    """上句是MP用的占位符，金标准类别的1热标签。"""
    input_sparse_gt = tf.placeholder(tf.float32, [config.BATCH_SIZE, num_rois, config.IMAGE_SHAPE[0]*config.projection_num], name="input_sparse_gt")
    # 上句，金标准复制了num_rois变，详见MaskRCNN_6_heads那个文件。
    return input_image, input_image_meta, input_rpn_match, input_rpn_bbox, \
           input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_1_hot, input_sparse_gt  # 返回一堆占位符，有的是略做了简单计算。


###################################################################################################################
#                                     填充占位符，准备在主函数中sess.run的时候用掉。                                   #
#                                     后续滑脱分级网络的填充占位符函数，也写在这儿了。                                  #
###################################################################################################################
def feed_placeholder(dataset, batch_size, given_ids, anchors, config, num_rois, input_image, input_image_meta, input_rpn_match,
                     input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_1_hot, input_sparse_gt, disp,
                     is_training):
    """
    占位符赋值。
    注意应该是在这里执行MaskRCNN_0_get_inputs.get_image_ids_for_next_batch这些函数，
        从dataset里得到这一批次的input_image_real、input_image_meta_real等。
    """
    if given_ids:  # 这种情况就是指定某两张图用来训练的，一般用于训练一两步检查网络的时候。
        assert len(given_ids) == batch_size
        image_id_selected = given_ids
        real_patient_id = []
        for i in image_id_selected:
            patiend_id_this = dataset.image_info[i]['patient']
            real_patient_id.append(patiend_id_this)
        real_patient_id = np.array(real_patient_id)
    else:
        image_id_selected, real_patient_id = MaskRCNN_0_get_inputs.get_image_ids_for_next_batch(dataset, batch_size,
                                                                                                shuffle=is_training)
        # 上句，复用网络的时候加了个shuffle=flip。
    if disp:
        print('当前已经完成的时代数：%d' % dataset._epochs_completed)
        print('当前批次开始的索引号：%d' % (dataset._index_in_epoch - batch_size))  # 现在dataset._index_in_epoch是当前批次结束的索引号，所以减一下。
        print('选择了以下这几个序号的图作为本批次的训练集：', image_id_selected, '真实病人序号是：', real_patient_id)
    # 上句，image_id_selected是0~119之间的，real_patient_id是实际病人序号。
    inputs_for_MRCNN_dict_feeding = MaskRCNN_0_get_inputs.get_batch_inputs_for_MaskRCNN(dataset, image_id_selected,
                                                                                        anchors, config, num_rois,
                                                                                        augment=is_training)
    input_image_real = inputs_for_MRCNN_dict_feeding[0]  # 归一化到0~255的uint8类型，考虑到这个类型确实是只能0~255，而且原来用MRCNN弄好了的那次也就是这样，就不弄到0~1了吧？。。
    input_image_meta_real = inputs_for_MRCNN_dict_feeding[1]
    input_rpn_match_real = inputs_for_MRCNN_dict_feeding[2]
    input_rpn_bbox_real = inputs_for_MRCNN_dict_feeding[3]
    input_gt_class_ids_real = inputs_for_MRCNN_dict_feeding[4]
    input_gt_boxes_real = inputs_for_MRCNN_dict_feeding[5]
    input_gt_masks_real = inputs_for_MRCNN_dict_feeding[6]  # MRCNN用的每个器官的掩膜
    input_gt_sparse_radon = inputs_for_MRCNN_dict_feeding[8]  # radon变换稀疏表示金标准
    gt_boxes_and_labels = np.concatenate([input_gt_boxes_real / 512, np.expand_dims(input_gt_class_ids_real, axis=2)],
                                         axis=2)
    gt_trim = message_passing_manners.batch_processing \
        (process_func=message_passing_manners.trim_gt, input_batch=gt_boxes_and_labels, num_classes=config.NUM_CLASSES)
    y_gt = gt_trim[:, :, 4]
    y_gt_one_hot_real = to_categorical(y_gt, config.NUM_CLASSES)  # 把标签变成1热形式。
    feed_dict = {input_image: input_image_real, input_image_meta: input_image_meta_real,
                 input_rpn_match: input_rpn_match_real, input_rpn_bbox: input_rpn_bbox_real,
                 input_gt_class_ids: input_gt_class_ids_real, input_gt_boxes: input_gt_boxes_real,
                 input_gt_masks: input_gt_masks_real, input_gt_1_hot: y_gt_one_hot_real, input_sparse_gt: input_gt_sparse_radon}
    # 上句中的input_gt_boxes: input_gt_boxes_real是说，赋给input_gt_boxes的是input_gt_boxes_real（还没有归一化的金标准外接矩形）
    """【基础】似乎sess.run和feed_dict的时候，不是所有的赋值都要用得上的。
    就像第一步run那个train_op_MRCNN，这一步不需要input_gt_mask，但是给feed进去也没啥的。。"""
    return feed_dict, image_id_selected, real_patient_id, input_image_real, \
           input_gt_class_ids_real, input_gt_boxes_real, input_gt_masks_real


########################################################################################################################
#                                                   以下是主函数。                                                       #
########################################################################################################################
def main(_):
    mode = 'test'
    assert mode in ['train', 'test']
    mode_dataset = 'split1'
    assert mode_dataset in ['random', 'split1', 'split2', 'split3', 'split4', 'split5']
    json_dir = '../data/detection_ver200204.json'  # 注意这个不是当前目录!!
    # 大电脑上，是E:/赵屾的文件/07-脊柱检测/程序/MaskRCNN_and_CRF/detection_ver190129.json
    # 小电脑上，是A:/pycharm_projects/19_MaskRCNN_and_CRF/detection_ver190129.json
    # 在服务器上，改成“/home/hnyz979/MaskRCNN_and_CRF/detection_ver190129.json”就可以用！！
    """注意有一个detection_ver190129_small.json，这是只有20张图的小数据集，按照下面的训练集-测试集分开，则会把这20个
    都放在训练集里。这个可以用来先测试一下网络能不能跑、损失函数有没有下降、用训练集弄的结果如何，这样可以看看是否在训练。"""
    file = open(json_dir, 'r', encoding='utf-8')
    s = json.load(file)
    file.close()
    dataset = make_dataset_from_json.get_MRCNN_json(s)
    if 'small' not in json_dir:
        dataset_train, dataset_test = splitting.get_train_test_sets_from_dataset(dataset, 5, mode_dataset)
        dataset_train_ids = MRCNN_utils.get_patient_ids(dataset_train)
        print('本次执行，训练集中的病人一共有%d个。' % len(dataset_train_ids), '索引号是：', dataset_train_ids)
        dataset_test_ids = MRCNN_utils.get_patient_ids(dataset_test)
        print('本次执行，测试集中的病人一共有%d个，' % len(dataset_test_ids), '索引号是：', dataset_test_ids)
    else:
        print('用小数据集调试程序。此数据集中，训练集和测试集是一样的，用来先测试一下梯度对不对、损失函数有没有下降、网络是否可以训练。')
        dataset_train, _ = splitting.get_train_test_sets_from_dataset(dataset, 5, mode_dataset)
        dataset_test = dataset_train
        dataset_train_ids = MRCNN_utils.get_patient_ids(dataset_train)
        print('本次执行，训练集中的病人一共有%d个。' % len(dataset_train_ids), '索引号是：', dataset_train_ids)
        dataset_test_ids = MRCNN_utils.get_patient_ids(dataset_test)
        print('本次执行，测试集中的病人一共有%d个，' % len(dataset_test_ids), '索引号是：', dataset_test_ids)
    if mode == 'train':
        train_flag = True  # 是否训练模式
        config = configure.Config()
        config.display()
        num_rois = config.TRAIN_ROIS_PER_IMAGE
        anchors = MaskRCNN_0_get_inputs.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                 config.RPN_ANCHOR_RATIOS,
                                                                 config.BACKBONE_SHAPES,
                                                                 config.BACKBONE_STRIDES,
                                                                 config.RPN_ANCHOR_STRIDE)
        # 上面生成锚。和class MaskRCNN()里第二步生成锚的那句一样。 shape=(65472, 4)。这个东西输出的anchors.dtype是dtype('float64')。
        anchors = anchors.astype('float32')  # 变成float32类型，因为后面MaskRCNN_4_Proposals里的一些东西需要float32。
        ################################################################################################################
        #                                              下面，构建MRCNN网络模型。                                          #
        ################################################################################################################
        class_numbers = config.NUM_CLASSES  # 注意，换数据集（如原来的分割数据集换成滑脱数据集），要到class_HaNDataset.py里去改掉这个config.NUM_CLASSES！！！
        input_image_placeholder, input_image_meta_placeholder, input_rpn_match_placeholder, input_rpn_bbox_placeholder, \
        input_gt_class_ids_placeholder, input_gt_boxes_placeholder, input_gt_masks_placeholder, input_gt_1_hot, input_sparse_gt \
            = get_placeholder(config, class_numbers, num_rois)
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
        """
        with tf.device('/cpu:0'):
            global_step = tf.train.create_global_step()
        注意，这个全局步数放这儿就不行了，因为这儿还没执行那个MaskRCNN_1_model.MaskMRCNN_model函数，所以全局步数会被变成第1个变量，
            即，后面所有变量就都往后错了一个，所以saver.restore(sess, checkpoint_path_keras)的时候就会报错，
            因为saver = tf.train.Saver({...})那句是按照第1个变量是MaskRCNN_1_model.MaskMRCNN_model里的第1个变量（即那个shape=(7, 7, 3, 64)的卷积核恢复的）
        又试了一下，即使用global_step = tf.Variable(0, trainable=False)这个定义，也是不行的。

        另外，加了这一句后，会发现这个global_step是一个变量，即<tf.Variable 'Variable:0' shape=() dtype=int32_ref>。
            而且，它被放在了网络变量的前面。不加这句的时候，是有690个变量，加了这个就成了691个了。
        """
        with tf.variable_scope("mask_rcnn_model"):
            """【基础】变量复用的“时效性”
            在结合GAN的时候，想把复用一下MaskRCNN_6_heads.build_fpn_mask_graph函数，但是：
                在外面main函数里加什么with tf.variable_scope，对于MaskRCNN_1_model.MaskMRCNN_model里的变量的复用是没有任何意义的。
                上面做法的意义只是，如果在main里、train模式下第二次用那个MaskRCNN_1_model.MaskMRCNN_model，
                    能保证第二次用的MaskRCNN_1_model.MaskMRCNN_model里的各个值，和第一次用的时候是一样的。但是它管不到MaskRCNN_1_model.MaskMRCNN_model里面的复用情况。
            所以试过这几种方法，都不行：
                1、如果tf.variable_scope(...)的()里什么都不加，那么MaskRCNN_1_model.MaskMRCNN_model里想要用两次同一个函数（比如说build_fpn_mask_graph）就不行。
                2、如果这儿加上reuse=True，那也报错，因为MaskMRCNN_model里的变量都还没定义呢，所以也不行。
                3、如果reuse=tf.AUTO_REUSE，构建网络的时候倒是不报错，但是看了一下程序里的变量名，发现多了12个变量，
                    就是那个build_fpn_mask_graph里的卷积层/反卷积层都重新弄了个变量，仍然没有达到效果。不过神奇的是，BN层都没有重新弄。
            所以只能：1、在MaskRCNN_1_model.MaskMRCNN_model里面的“mrcnn_mask = MaskRCNN_6_heads.build_fpn_mask_graph...”那句，加上withtf.variable_scope(...)什么的，
                而且：2、要把MaskRCNN_6_heads.build_fpn_mask_graph的卷积层什么的写成tf.get_variable的形式（类似于MaskRCNN_3_RPN.py），
                当然这样的话，变量名就变掉了，那个matching_ckpt_variable_names_to_program_variable_names里就要跟着修改。
            """
            rpn_class, rpn_bbox, mrcnn_class, mrcnn_bbox, mrcnn_mask, lspmr_loss, rpn_class_loss, rpn_bbox_loss, \
            class_loss, bbox_loss, mask_loss, sparse_loss, sparsity_loss, total_loss_MRCNN, mrcnn_boxes_and_scores, mrcnn_boxes_and_scores_sorted = \
                MaskRCNN_1_model.MaskMRCNN_model('training', config, global_step, anchors, input_image_placeholder,
                                                 input_image_meta_placeholder,
                                                 input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                                 input_gt_class_ids_placeholder, input_gt_boxes_placeholder,
                                                 input_gt_masks_placeholder, input_sparse_gt, train_flag)
            # 另外，这个地方设断点停住，输入all_variables = tf.global_variables()，可以观察现在有多少个变量，以及他们的名字都叫什么。。
        ################################################################################################################
        #                                  下面，构建信息传递，把它和前面的MRCNN网络整合到一起去。                           #
        ################################################################################################################
        mrcnn_boxes_and_scores_full = tf.concat([mrcnn_boxes_and_scores, mrcnn_class], axis=2)  # shape=(2, 60, 16)
        mrcnn_threshold = config.DETECTION_MIN_CONFIDENCE_MP * tf.ones([mrcnn_boxes_and_scores_full.shape[0]],
                                                                       dtype=tf.float32)
        num_classes = config.NUM_CLASSES * tf.ones([mrcnn_boxes_and_scores_full.shape[0]], dtype=tf.int32)
        mrcnn_boxes_and_scores_sorted1 = MRCNN_utils.batch_slice \
            ([mrcnn_boxes_and_scores_full, mrcnn_boxes_and_scores_full[:, :, 4], mrcnn_threshold, num_classes],
             lambda x, y, z, a: MRCNN_utils.select_high_scores_for_mp(x, y, z, a),
             config.IMAGES_PER_GPU, names="m")  # <tf.Tensor 'm:0' shape=(2, ?, ?) dtype=float32>，应该是shape=[2, 9, 16]
        # 上句其实有点不规范，是选出来了类别大于0.9的东西，就是非背景类，而，实际上不是选出来信心分值。
        y_pred = mrcnn_boxes_and_scores_sorted1[:, :,
                 6:]  # 应该是shape=[2, 9, 10]、即[config.BATCH_SIZE, config.NUM_CLASSES-1, config.NUM_CLASSES]
        y_pred.set_shape([config.BATCH_SIZE, config.NUM_CLASSES - 1, config.NUM_CLASSES])
        # 以上，构建金标准标签和预测值标签的占位符。第二个config.NUM_CLASSES-1是每张图最多检测到的结果数。
        #     因为现在是训练集，所以每张图最多就检测到9个（因为最多9个金标准）
        use_trained_Psi = False
        if use_trained_Psi:
            _MP_mat_dir = './Phi_and_Psi_mat_' + mode_dataset + '.mat'  #
            _Phi_mat = sio.loadmat(_MP_mat_dir)['Phi']
            _Psi_mat = sio.loadmat(_MP_mat_dir)['Psi']
            _global_step_MP = int(sio.loadmat(_MP_mat_dir)['GS'])  # 必须变成int，即让shape=()，否则是shape=[1,1]，后面会报错。
            Phi_mat = tf.Variable(_Phi_mat, name='MP_Phi')
            Psi_mat = tf.Variable(_Psi_mat, name='MP_Psi')
            global_step_MP = tf.Variable(_global_step_MP, name='global_step', trainable=False)  #
            print('用训练好的Phi和Psi矩阵。')
        else:
            Phi_mat = tf.get_variable("MP_Phi", [config.NUM_CLASSES, config.NUM_CLASSES],
                                      initializer=tf.contrib.keras.initializers.he_normal())
            # 注意，上句的Phi_mat应该和下句的Psi_mat的shape一般是不一样的，但是我们这儿是一样的，因为我们的CPV也是10个元素的。
            Psi_mat = tf.get_variable("MP_Psi", [config.NUM_CLASSES, config.NUM_CLASSES],
                                      initializer=tf.contrib.keras.initializers.he_normal())
            global_step_MP = tf.Variable(0, name='global_step', trainable=False)  # 初始化为0，然后自增长。增长到decay_steps就会降低学习率。
            print('随机初始化Phi和Psi矩阵。')
        y_pred_calibrated = message_passing_manners.get_calibrated_CPV(y_pred, Phi_mat,
                                                                       Psi_mat)  # 这个是(批大小. 9, 10)的没问题，问题是，金标准怎么对上啊？
        yt = tf.argmax(input_gt_1_hot, axis=2)  # 应该是就把那个1热标签变成一个数的标签就行了。
        MP_loss = 0
        correct_batch = []
        for i in range(config.BATCH_SIZE):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits \
                (logits=y_pred_calibrated[i, :, :], labels=input_gt_1_hot[i, :, :], name='xentropy')
            _, nz = MRCNN_utils.trim_zeros_graph(y_pred[i, :, :], name='n')  # 非补0的位置（用y_pred来看，因为这个是知道有补零的）
            valid_sum = tf.reduce_sum(tf.cast(nz, tf.float32))  # 有多少个非补零的？
            cross_entropy_valid = tf.boolean_mask(cross_entropy, nz)  # 非补零位置的交叉熵
            loss_this = tf.reduce_mean(cross_entropy_valid, name='xentropy_mean')
            MP_loss += loss_this
            y_pred_calibrated_this = tf.boolean_mask(y_pred_calibrated[i, :, :], nz)  # 非补零位置的CPV
            positive_roi_ix = tf.where(yt[i, :] > 0)[:, 0]  # 非补0的位置
            yt_valid_this = tf.gather(yt[i, :], positive_roi_ix)  # 非补零位置的金标准标签
            correct = tf.nn.in_top_k(y_pred_calibrated_this, yt_valid_this, 1)  # 非补零位置，对的得分就是True，否则是False。
            mean_correct_num = tf.divide(tf.reduce_sum(tf.cast(correct, tf.int32)), tf.cast(valid_sum, tf.int32))
            correct_batch.append(mean_correct_num)
        correct_batch_mean = tf.reduce_mean(correct_batch)  # 这一批次内的平均正确率。【基础】两个张量组成的list，可以直接平均。
        if config.MP_ignore_first_label:
            Psi_mat_valid = Psi_mat[1:, 1:]  # 把第0行和第0列删掉
        else:
            Psi_mat_valid = Psi_mat
        MP_loss_path = message_passing_manners.path_loss(input_gt_1_hot, y_pred_calibrated, Psi_mat_valid,
                                                         ignore_first_label=config.MP_ignore_first_label)  # 路径损失
        MP_loss_total = MP_loss + MP_loss_path
        ################################################################################################################
        #                                   下面考虑变量初始化问题。想分为初始化滑脱和不初始化滑脱两种情况。                   #
        ################################################################################################################
        all_vars = tf.all_variables()  # 应该是程序中、加入了GAN网络之后的所有变量。含有MRCNN的变量、GAN的变量、全局步数、（后来加上的）滑脱分级的变量。
        print("程序里的变量名（一共%d个）：" % len(all_vars))  # 如果用BN的话，应该是690多个或者710多个；如果用GN的话，就是507个了（因为moving_mean那些没了）。
        print("他们是：", all_vars)
        t_vars = tf.trainable_variables()  # 所有可训练的参数，应该有MRCNN的那些和GAN的那些，除了那个全局步数之外应该都在里面。
        g_vars_trainable = [var for var in t_vars if 'mask_rcnn_model' in var.name]  # 类似地，G网络的参数。不太确定。。
        MRCNN_vars = [var for var in all_vars if 'mask_rcnn_model' in var.name]  # 原来MRCNN的变量
        """总结一下程序执行到上句，所有变量的情况：
        all_vars：一共711个，第0个是全局步数那样的，第1~690个是MRCNN的（一共690个），第691个似乎又是个全局变量，第692~710个是GAN的（一共19个）。
                GAN修改成输出一个数的之后，变成了715个。
        t_vars：一共483个，似乎是没有那些moving_mean和moving_variance。其中MRCNN网络中的变量都是mask_rcnn_model/...开头的，GAN网络的变量都是d_gan/开头的。
        g_vars_trainable：一共470个，就是MRCNN里的可训练变量。发现470+13正好是483，说明所有可训练的变量都包括进去了。
        """
        checkpoint_path_keras = '../checkpoints_from_keras/results'  # 不是当前目录！！
        # 大电脑： 'E:/赵屾的文件/06-脊柱滑脱/Spine-Gan-plus-MRCNN/tmp/checkpoints_from_keras/results'
        # 小电脑： 'A:/pycharm_projects/14_2Spine-Gan-plus-MRCNN/tmp/checkpoints_from_keras/results'
        # 服务器： '/home/hnyz979/Spine-Gan-plus-MRCNN/tmp/checkpoints_from_keras/results'
        path_keras = pathlib.Path(checkpoint_path_keras + '.meta')
        sess = tf.Session()
        if config.FEN_network == 'Resnet101':  #
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-Resnet101-mp-' + mode_dataset + '-dict-40002'  # 【基础】当前目录就是./。
        elif config.FEN_network == "Resnet50":  #
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-Resnet50-20000'
        elif config.FEN_network == "VGG19":
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-VGG19-20000'
        elif config.FEN_network == "VGG19_FCN":
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-VGG19_FCN-20000'
        elif config.FEN_network == "VGG19_FCN_simp":
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-VGG19_FCN_simp-20000'
        elif config.FEN_network == "Googlenet":
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-Googlenet-20000'
        else:  # 就是Densenet的情况
            checkpoint_MRCNN = './tmp/要保留的结果/MRCNN-Densenet-20000'
        path_MRCNN = pathlib.Path(checkpoint_MRCNN + '.meta')  # 检测是否存在文件
        saver_MRCNN = tf.train.Saver(MRCNN_vars)  # 【基础】这个saver_MRCNN，只存/取MRCNN的那些变量。
        """
        【变量初始化】
        上面的这个saver，是“管理（保存或者读取）”所有变量的。
        后面的那个saver_first_train，是“管理（保存或者读取）”一部分（就是()里定义的那些）变量的。
            注意，最早用MRCNN做不同的器官分类，就是那个MaskRCNN_aux0...的，后来想用来分割L4-L5-S1，就要用MaskRCNN_aux1...了，
            因为如果还用MaskRCNN_aux0...，会因为有的变量形状不对而报错。
            然而，MaskRCNN_aux1并没有初始化所有的变量，initialize_uninitialized程序会把未初始化的初始化掉。
        再那个saver_MRCNN_GAN，是因为后面又加了滑脱分级网络，而，一开始的时候，我显然是希望先用已有的ckpt文件把MRCNN和GAN恢复了，
            再去训练那个滑脱分级的。
        """
        ckpt_used = 'keras'
        assert ckpt_used in ['None', 'keras', 'Mask']  #
        if ckpt_used == 'keras':
            assert config.FEN_network in ["Resnet101", "Resnet50"], 'keras变过来的节点文件，只有Resnet101和Resnet50这两种。'
            if config.FEN_network == "Resnet101":
                saver_first_train = MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis. \
                    matching_ckpt_variable_names_to_program_variable_names()  # 用resnet101来分割L4-L5-S1的
            else:
                saver_first_train = MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis. \
                    matching_ckpt_variable_names_to_program_variable_names_comp()  # 用resnet50来分割L4-L5-S1的
            if path_keras.is_file():  # 如果存在文件
                print("在以下路径发现了ckpt文件：", checkpoint_path_keras)
                checkpoints = cp.list_variables(checkpoint_path_keras)  # 注意这儿要用checkpoint_path而不是path。
                print("节点文件中的变量名（一共%d个）：" % len(checkpoints))
                print(checkpoints)  # 这儿打印出来的是节点文件中的变量名
                # global_step = tf.Variable(0, trainable=False)
                # 上面那句注释：如果是从keras里读出来节点文件，那么仍然认为还没有用tf训练，所以记为0步。但由于前面已经设置了全局步数=0，所以可以注释掉了。
                # 但是，这种情况下由于没有初始化global_step，所以是会报错的，所以加了下面一句sess.run(global_step.initializer)。
                #     那个else里的情况，是因为所有变量都统一初始化了，所以反而不需要单独初始化global_step。
                sess.run(global_step.initializer)
                saver_first_train.restore(sess, checkpoint_path_keras)
                print("从keras变过来的节点文件中恢复了模型。")
                MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis.initialize_uninitialized(sess)
                print("初始化了未恢复的变量。")
                # 上句，初始化掉未恢复的变量。
            else:
                # global_step = tf.Variable(0, trainable=False)   # 类似地，全局步数设为0；而由于前面已经设了，所以这儿可以注释掉。
                init = tf.initialize_all_variables()
                sess.run(init)
                print('没有发现keras变过来的节点文件，从头开始训练。')
        elif ckpt_used == 'Mask':
            if path_MRCNN.is_file():
                print("在以下路径发现了ckpt文件：", checkpoint_MRCNN)
                checkpoints = cp.list_variables(checkpoint_MRCNN)  # 注意这儿要用checkpoint_path而不是path。
                print("节点文件中的变量名（一共%d个）：" % len(checkpoints))
                print(checkpoints)  # 这儿打印出来的是节点文件中的变量名
                global_step1 = checkpoint_MRCNN.split('/')[-1].split('-')[-1]  # 这个地方把全局步数设成“该节点文件已进行了的步数”。
                global_step1 = int(global_step1)
                global_step = tf.Variable(global_step1, trainable=False)  # 好像得给他弄成这个形式。。
                saver_MRCNN.restore(sess, checkpoint_MRCNN)  # 节点是掩膜和GAN的变量，但是只恢复MRCNN部分。
                print("从本程序以前训练得到的节点文件中恢复了MRCNN模型。")
                sess.run(global_step.initializer)
            else:
                # global_step = tf.Variable(0, trainable=False)   # 如果没有ckpt文件，那么全局步数设为0；而由于前面已经设了，所以这儿可以注释掉。
                init = tf.initialize_all_variables()
                sess.run(init)
                print('没有发现以前的节点文件，从头开始训练。')
                """
                【基础】【重要】节点文件、全局步数、训练操作和初始化
                通过读取节点文件，可以实现变量初始化的功能，所以就不需要再弄那个init了。当然这个节点文件因为是从keras里弄的，所以变量名和
                    本程序里的不一样，就有点麻烦，一般人家就直接saver = tf.train.Saver() 就完了，而我得在那个()里把所有变量名对应起来，
                    为此还特别弄了一个函数。当然，这个saver弄好了之后，都是用saver.restore(sess, 节点文件路径)去恢复。详见“【基础】关于saver”。
                然后，发现我这个节点文件里没有保存全局步数，所以这个全局步数需要单独设定、单独初始化。
                    通过文件名能够得到已经执行了的步数（如MRCNN_tf-9999就是执行了9999步的），赋值给global_step，
                    这里需要注意必须给他变成tf.Variable才行，否则如果是str或者int类型的数据，都会报错（不能放到后续构造train_op的函数里）。
                    然后用sess.run(global_step.initializer)把它单独初始化掉。
                step和global_step的关系：step是这一次执行程序训练的步数，而global_step是一共已经训练了的步数。比如说，如果我恢复了那个已经
                    训练了9999步的节点文件，来执行一次程序，那么一开始step就是0，而global_step则是9999，然后构造那个train_op的时候变成了10000，
                    即从10000开始。
                最后，构造训练操作train_op。
                    按照这里说的：https://blog.csdn.net/chenxicx1992/article/details/56483180
                    自己定义的optimizer会生成新的variables，但是并没有初始化，所以直接用回报错掉。
                    所以就只好在这两层的if-else-if-else语句里，这四种情况下都把已经初始化（或者恢复）的所有变量保存在initialized_vars里，
                    然后在if-else-if-else语句外面构造train_op，然后弄出来所有的没初始化的变量，再定义那个init1把它初始化掉。
                验证了一下学习率，如果训练了10000步，初始学习率是0.001的话，那就是0.001*0.96^(10000/42)=6.0099*10^-8，
                    而实际上算出来的是6.033248e-08，说明学习率算得应该没问题的。然后也看了下，学习率确实是在5个时代
                    （因为config.num_epochs_per_decay设的是5，真干的时候应该要设大一些吧）后就变一次的。
                总之，1，恢复的时候变量名要对应；2，节点文件里没有的可以单独初始化；3，全局步数要设为tf.Variable；
                    4，在构造train_op的时候把全局步数放到optimizer.minimize/apply_gradients也可以实现全局步数自增操作；
                    5，网络变量恢复了，但是train_op如果构造了新的变量，可以弄出来没初始化的那些变量。
                """
        else:
            init = tf.initialize_all_variables()
            sess.run(init)
            print('从头开始训练。')
        MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis.initialize_uninitialized(sess)
        print("初始化了未恢复的变量。")
        """
        上句，可以测试是否还有变量没初始化，并且初始化掉他们。
        第一次跑的时候，GAN的所有变量应该都是没用预训练过的，所以第一次应该是会在那个initialize_uninitialized函数里把
            GAN网络的所有变量都打印出来（当然还应该打印出来MRCNN里的6个没初始化的变量），然后初始化掉。
        后面再跑的时候，因为上一次训练得到的变量，应该已经用saver把这个程序里的所有变量存到ckpt文件里了，所以，理论上
            那个initialize_uninitialized函数应该不会打印出来任何的变量名。
        """
        ################################################################################################################
        #                    下面构建训练操作train_op，分两个来，一个是train_op_MRCNN，一个是train_op_MP。                  #
        ################################################################################################################
        train_op_MRCNN, configured_learning_rate = MaskRCNN_1_model.train_MaskMRCNN_model \
            (config, g_vars_trainable, len(dataset_train.image_info), total_loss_MRCNN,
             global_step)  # 不要GAN的，就把G_loss改回total_loss_MRCNN
        """上句，用train_MaskMRCNN_model定义MRCNN网络的训练操作。
        另外，上句执行后，用all_vars1 = tf.all_variables()和len(all_vars1)就可以查看变量个数，然后就发现变量数量几乎翻了两倍。
        原来是508个（比前面打印出来的507个多了1个，先不管他），执行完了上句就变成978个，
            然后用set(all_vars1)-set(all_vars)看了一下，
            发现关于MRCNN的所有变量（就是变量名以“mask_rcnn_model/...”开始的）的最后都加了个/Momentum:0项，
            估计就是构造那个train_op_MRCNN的时候给加的动量项。
        类似地，构造了train_op_GAN之后，变量数又会变多，就变成了1012个。
        """
        MP_vars = [var for var in all_vars if 'MP' in var.name]
        decay_steps_MP = config.decay_steps_MP
        learning_rate_decay_factor_MP = config.learning_rate_decay_factor_MP
        configured_learning_rate_MP = tf.train.exponential_decay(config.LEARNING_RATE_MP, global_step_MP,
                                                                 decay_steps_MP,
                                                                 learning_rate_decay_factor_MP, staircase=True,
                                                                 name='exponential_decay_lr_MP')
        optimizer_MP = tf.train.GradientDescentOptimizer(configured_learning_rate_MP)
        grads_MP = optimizer_MP.compute_gradients(MP_loss_total, var_list=MP_vars)  # MP_loss
        train_op_MP = optimizer_MP.apply_gradients(grads_MP, global_step=global_step_MP)
        # 以上，构建训练图、训练操作。
        ################################################################################################################
        #                                             下面，开始训练MRCNN网络。                                          #
        ################################################################################################################
        MaskRCNN_aux1_get_saver_to_restore_ckpt_spondylolisthesis.initialize_uninitialized(sess)
        # 上句是初始化所有未初始化的变量。因为定义那个train_op_MRCNN的时候，为了定义那个动量项，会让变量数增加一倍（具体见上面）。
        max_steps_MRCNN = config.TRAIN_STEP
        save_dir = './tmp/MRCNN-' + config.FEN_network + '-mp-' + mode_dataset
        if config.USE_LISTA:
            save_dir += '-LISTA'
        for step in range(max_steps_MRCNN):  # 先训练100步试试，不管他多少个时代了。。
            feed_dict, id, real_patient_id, _, _, _, _ = feed_placeholder(dataset_train, config.BATCH_SIZE, [], anchors, config, num_rois,
                                                           input_image_placeholder, input_image_meta_placeholder,
                                                           input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                                           input_gt_class_ids_placeholder, input_gt_boxes_placeholder,
                                                           input_gt_masks_placeholder,
                                                           input_gt_1_hot, input_sparse_gt, disp=True, is_training=True)
            try:
                # graph = tf.get_default_graph()
                # rpn_rois = graph.get_tensor_by_name('mask_rcnn_model/packed_2:0')
                # sparse = graph.get_tensor_by_name('mask_rcnn_model/LISTA_dict_learning/transpose_2:0')
                # target_sparse = graph.get_tensor_by_name('mask_rcnn_model/target_sparse:0')
                # roi = graph.get_tensor_by_name('mask_rcnn_model/rois:0')
                #
                # rpn_rois_val, sparse_val, target_sparse_val, roi_val = sess.run([rpn_rois, sparse, target_sparse, roi], feed_dict=feed_dict)

                _, total_loss_val, lspmr_loss_val, rpn_class_loss_val, rpn_bbox_loss_val, class_loss_val, bbox_loss_val, \
                mask_loss_val, sparse_loss_val, sparsity_loss_val, _global_step, _configured_learning_rate\
                    = sess.run(
                    [train_op_MRCNN, total_loss_MRCNN, lspmr_loss, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss,
                     mask_loss, sparse_loss, sparsity_loss, global_step, configured_learning_rate], feed_dict=feed_dict)

                """
                【基础】如何检查程序里的变量是否有错误：
                在上一版的程序里，很傻地把想检查的变量都返回回来，run掉，然后检查，其实不需要的，只用把那个变量名弄出来就行了。
                比如：
                    MaskRCNN_1_model.py里，用了那个DetectionTargetLayer函数，然后输出的target_class_ids是这样的:
                    <tf.Tensor 'mask_rcnn_model/target_class_ids:0' shape=(4/2, ?) dtype=int32>
                    然后，在主函数上面的sess.run那块设断点，输入：
                    graph = tf.get_default_graph()
                    target_class_ids=graph.get_tensor_by_name('mask_rcnn_model/target_class_ids:0')
                    _target_class_ids = sess.run(target_class_ids, feed_dict=feed_dict)
                这就相当于是把那个target_class_ids给run掉了，这样就不用返回那么多乱七八糟的东西了。
                """
            except:
                print('发生异常的索引号：', id, '真实病人索引号：', real_patient_id)
                print(traceback.format_exc())  # 如果不对，打印出来问题，继续执行。
                total_loss_val = 10000.
                lspmr_loss_val = 10000.
                rpn_class_loss_val = rpn_bbox_loss_val = 10000.
                class_loss_val = bbox_loss_val = mask_loss_val = sparse_loss_val = sparsity_loss_val = 10000.
                _configured_learning_rate = 1.

            """
            【基础】如何检查程序里的变量是否有错误：
            在上一版的程序里，很傻地把想检查的变量都返回回来，run掉，然后检查，其实不需要的，只用把那个变量名弄出来就行了。
            比如：
                MaskRCNN_1_model.py里，用了那个DetectionTargetLayer函数，然后输出的target_class_ids是这样的:
                <tf.Tensor 'mask_rcnn_model/target_class_ids:0' shape=(4/2, ?) dtype=int32>
                然后，在主函数上面的sess.run那块设断点，输入：
                graph = tf.get_default_graph()
                target_class_ids=graph.get_tensor_by_name('mask_rcnn_model/target_class_ids:0')
                _target_class_ids = sess.run(target_class_ids, feed_dict=feed_dict)
            这就相当于是把那个target_class_ids给run掉了，这样就不用返回那么多乱七八糟的东西了。
            """


            if step % 1 == 0:
                print('第%d步（全局步数为%d），各种损失情况: total_loss_MRCNN = %.5f, lspmr_loss = %.5f, rpn_class_loss = %.5f, \n'
                      'rpn_bbox_loss = %.5f, class_loss = %.5f, bbox_loss = %.5f, mask_loss = %.5f, '
                      'sparse_loss=%.5f, sparsity_loss=%.5f, 当前MRCNN网络学习率是：%.10f。' %
                      (step, _global_step, total_loss_val, lspmr_loss_val, rpn_class_loss_val, rpn_bbox_loss_val,
                       class_loss_val, bbox_loss_val, mask_loss_val, sparse_loss_val, sparsity_loss_val,
                        _configured_learning_rate))

            if (step + 1) == max_steps_MRCNN:  # 暂时去掉，不保存了    or (step + 1) == 6000
                saver_MRCNN.save(sess, save_dir, global_step=_global_step)
        print("MRCNN训练%d次，暂停训练MRCNN，开始训练MP。" % max_steps_MRCNN)
        ################################################################################################################
        #                               下面，存一下训练的MRCNN结果，即输入信息传递的结果。                                  #
        #                   注意，这个存的是有补零的、训练结果和对应的金标准已经对应好了的、尚未做信息传递的、训练结果。           #
        ################################################################################################################
        # CPVs_all_train = np.zeros([dataset_train_ids.shape[0], config.NUM_CLASSES - 1, config.NUM_CLASSES])  # 训练集、MP前的CPV
        # CPVs_all_train_MP = np.zeros([dataset_train_ids.shape[0], config.NUM_CLASSES - 1, config.NUM_CLASSES])  # 训练集、MP后的CPV
        # gts_for_MP = np.zeros([dataset_train_ids.shape[0], config.NUM_CLASSES - 1, config.NUM_CLASSES])
        # feed_steps = int(dataset_train_ids.shape[0] / config.BATCH_SIZE)
        # for i in range(feed_steps):
        #     feed_dict2, _, _, _, _, _, _ = feed_placeholder \
        #         (dataset_train, config.BATCH_SIZE, [], anchors, config,
        #          input_image_placeholder, input_image_meta_placeholder, input_rpn_match_placeholder,
        #          input_rpn_bbox_placeholder, input_gt_class_ids_placeholder, input_gt_boxes_placeholder,
        #          input_gt_masks_placeholder, input_gt_1_hot, disp=False, is_training=False)  # 此处不训练、不乱序
        #     _y_pred, _y_pred_calibrated, _input_gt_1_hot = sess.run([y_pred, y_pred_calibrated, input_gt_1_hot], feed_dict=feed_dict2)
        #     CPVs_all_train[2*i:2*i+2, :_y_pred.shape[1], :] = _y_pred
        #     CPVs_all_train_MP[2*i:2*i+2, :_y_pred_calibrated.shape[1], :] = _y_pred_calibrated
        #     gts_for_MP[2*i:2*i+2, :input_gt_1_hot.shape[1], :] = _input_gt_1_hot
        # sa = 'CPV_train_before_MP.mat'  #
        # sio.savemat(sa, {'CPVs': CPVs_all_train, 'gts': gts_for_MP})  # 存一下训练集中、输入信息传递的东西。
        # sa1 = 'CPV_train_MP.mat'  #
        # sio.savemat(sa1, {'CPVs': CPVs_all_train_MP})  # 存一下训练集中、信息传递输出的东西。
        ################################################################################################################
        #                                                   下面训练MP                                                  #
        ################################################################################################################
        _global_step_MP = 0
        for step_MP in range(config.TRAIN_STEP_MP):
            feed_dict1, id, real_patient_id, _, _, _, _ = feed_placeholder(dataset_train, config.BATCH_SIZE, [], anchors, config, num_rois,
                                                           input_image_placeholder, input_image_meta_placeholder,
                                                           input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                                           input_gt_class_ids_placeholder, input_gt_boxes_placeholder,
                                                           input_gt_masks_placeholder,
                                                           input_gt_1_hot, input_sparse_gt, disp=False, is_training=False)
            try:
                _, _MP_loss, _MP_loss_path, _global_step_MP, _configured_learning_rate_MP = sess.run \
                    ([train_op_MP, MP_loss, MP_loss_path, global_step_MP, configured_learning_rate_MP],
                     feed_dict=feed_dict1)
            except:
                print('发生异常的索引号：', id, '真实病人索引号：', real_patient_id)
                graph = tf.get_default_graph()
                pad1 = graph.get_tensor_by_name('Pad_1:0')
                pad0 = graph.get_tensor_by_name('Pad:0')
                _pad0, _pad1 = sess.run([pad0, pad1], feed_dict=feed_dict1)
                # 【分析】有一次run的时候报错，说是batch_slice里有问题，两个拼接的张量长度不相等，所以run掉的时候报错。
                # 但是，即使一直用同样的两张图去做（split3的268和349），这个报错也是有的时候出现、有的时候不出现。
                # 考虑到训练的时候确实会有一定的随机性（两次执行程序，一样的输入，输出有差别），如果出错就跳过，不会影响程序训练结果。
                _MP_loss = 10000
                _MP_loss_path = 10000
                _configured_learning_rate_MP = 0
            if step_MP % 100 == 0:
                print('信息传递训练步数：', step_MP, '信息传递全局步数：', _global_step_MP, 'MP训练损失：', _MP_loss,
                      'MP训练路径损失：', _MP_loss_path, '信息传递学习率：', _configured_learning_rate_MP)
        print('训练完，看一下转移矩阵是什么样子。')
        _Phi_mat, _Psi_mat, _global_step_MP = sess.run([Phi_mat, Psi_mat, global_step_MP])  # 这儿应该不需要feed_dict吧。。
        print(_Psi_mat)
        # _Psi_mat[0, :] = 0
        # _Psi_mat[:, 0] = 0
        # 上句，Phi_and_Psi_mat_split1，因为_Psi_mat里没有忽略BG标签，现在就把它们的权重都置零。
        savename4 = './tmp/Phi_and_Psi_mat_' + mode_dataset + '.mat'
        sio.savemat(savename4, {'Psi': _Psi_mat, 'Phi': _Phi_mat, 'GS': _global_step_MP})  # 保存，慎重使用，别把以前存的给覆盖了。
        ################################################################################################################
        #                              下面训练自适应SG滤波，即，找到训练集中，滤波前后差别的最大值                           #
        ################################################################################################################
        diff_max = 0
        feed_steps = int(dataset_train_ids.shape[0] / config.BATCH_SIZE)
        for st in range(feed_steps):
            feed_dict3, id, real_patient_id, _, _, _, _ = feed_placeholder(dataset_train, config.BATCH_SIZE, [], anchors, config, num_rois,
                                                           input_image_placeholder, input_image_meta_placeholder,
                                                           input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                                           input_gt_class_ids_placeholder, input_gt_boxes_placeholder,
                                                           input_gt_masks_placeholder,
                                                           input_gt_1_hot, input_sparse_gt, disp=False, is_training=False)
            try:
                _mrcnn_boxes_and_scores_sorted1 = sess.run(mrcnn_boxes_and_scores_sorted1, feed_dict=feed_dict3)
                for i in range(config.BATCH_SIZE):
                    x1 = _mrcnn_boxes_and_scores_sorted1[i, :, 1]  # 结果的x1坐标
                    ix_unpadded = np.where(x1 > 0)[0]  # 非补0的位置
                    x1_trimmed = x1[ix_unpadded]
                    l = len(x1_trimmed)
                    if l % 2 == 1:
                        window_length = l
                    else:
                        window_length = l + 1
                    x1_trimmed_f = savgol_filter(x1_trimmed, window_length=window_length, polyorder=2, mode='mirror')
                    """上句，关于SG滤波器，详见《A57_savitzky_golay.py》。"""
                    diff = abs(x1_trimmed - x1_trimmed_f)
                    print('训练集，第%d张图，滤波误差最大是%.5f' % (real_patient_id[i], np.max(diff)))
                    if np.max(diff) > diff_max:
                        diff_max = np.max(diff)
            except:
                print('发生异常的索引号：', id, '真实病人索引号：', real_patient_id)
                print(traceback.format_exc())  # 如果不对，打印出来问题，继续执行。这个问题其实很可能跟上面训练信息传递的时候一样的。
                pass
        print('训练集中，滤波后最大的相差是%.5f' % diff_max)
        savename5 = './tmp/SG_threshold_' + mode_dataset + '.mat'
        sio.savemat(savename5, {'diff_max': diff_max})  # 保存，慎重使用，别把以前存的给覆盖了。
        print('训练结束。')

    if mode == 'test':
        train_flag = True  # 本来以为是不是BN的问题，如果测试的时候应该把那个is_training改成False，结果一看改了更差。。。
        """
        测试部分的编程思路：
        前面和训练部分都一样，弄数据集、占位符，然后用占位符去执行MaskRCNN_1_model.MaskMRCNN_model函数，只不过这个时候用的是inference，
            所以输出和训练部分不一样（尽管输入是一样的）。
        然后把原图用get_batch_inputs_for_MaskRCNN得到赋值占位符的东西feed_dict，
            用它把detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rois, rpn_class, rpn_bbox这些东西给run掉，就算是执行完了。
        （前面这些和keras程序不一样的）
        最后就是把那个detections用unmold_detections得到最终结果，然后用visualize.display_instances展示出来。（这些和keras里的是一样的）
        """
        config = configure.Config()
        config.DETECTION_MIN_CONFIDENCE = 0.8  # 测试的时候，这个东西稍微减小些。。
        config.display()
        ################################################################################################################
        #                                        下面，恢复信息传递矩阵，及其前面的滤波阈值                                 #
        ################################################################################################################
        _MP_mat_dir = './tmp/Phi_and_Psi_mat_' + mode_dataset + '.mat'  #
        _Phi_mat = sio.loadmat(_MP_mat_dir)['Phi']
        _Psi_mat = sio.loadmat(_MP_mat_dir)['Psi']
        Phi_mat = tf.constant(_Phi_mat, dtype=tf.float32)  # 测试的时候不需要改变它的值，所以用constant。
        Psi_mat = tf.constant(_Psi_mat, dtype=tf.float32)  # 测试的时候不需要改变它的值，所以用constant。
        class_id_max = config.NUM_CLASSES - 1  # 最大类别序号是类别数-1
        class_id_min = 1  # 最小类别序号是1（背景类是0）
        _SG_thre_dir = './tmp/SG_threshold_' + mode_dataset + '.mat'
        _SG_thre = sio.loadmat(_SG_thre_dir)['diff_max'] + 0.01  # 加个0.01的裕度吧，这个裕度随便写的。
        _SG_thre = float(_SG_thre)  # 必须变成float，否则他是一个(1,1)的东西，而不是一个数（shape=()）。
        ################################################################################################################
        #                                              以下，为viterbi修正做准备                                         #
        ################################################################################################################
        letterdict = {'0BG': 0, '0S1': 1, '0L5': 2, '0L4': 3, '0L3': 4, '0L2': 5, '0L1': 6, 'T12': 7, 'T11': 8,
                      'T10': 9}
        # letterdict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        trans = {}
        for i in ['0BG', '0S1', '0L5', '0L4', '0L3', '0L2', '0L1', 'T12', 'T11', 'T10']:
            # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
            for j in ['0BG', '0S1', '0L5', '0L4', '0L3', '0L2', '0L1', 'T12', 'T11', 'T10']:
                # '0BG', '0S1', '0L5', '0L4', '0L3', '0L2', '0L1', 'T12', 'T11', 'T10'
                trans[i + j] = _Psi_mat[letterdict[i], letterdict[j]]  # trans在这儿是个字典，有16个内容，就是各种标签的转移路径得分
        ################################################################################################################
        #                                               以下，恢复MRCNN的变量                                            #
        ################################################################################################################
        anchors = MaskRCNN_0_get_inputs.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                 config.RPN_ANCHOR_RATIOS,
                                                                 config.BACKBONE_SHAPES,
                                                                 config.BACKBONE_STRIDES,
                                                                 config.RPN_ANCHOR_STRIDE)
        # 上面生成锚。和class MaskRCNN()里第二步生成锚的那句一样。 shape=(65472, 4)。这个东西输出的anchors.dtype是dtype('float64')。
        anchors = anchors.astype('float32')  # 变成float32类型，因为后面MaskRCNN_4_Proposals里的一些东西需要float32。
        class_numbers = config.NUM_CLASSES
        num_rois = config.POST_NMS_ROIS_INFERENCE
        input_image_placeholder, input_image_meta_placeholder, input_rpn_match_placeholder, input_rpn_bbox_placeholder, \
        input_gt_class_ids_placeholder, input_gt_boxes_placeholder, input_gt_masks_placeholder, input_gt_1_hot, input_sparse_gt \
            = get_placeholder(config, class_numbers, num_rois)  # 这个地方就把各个占位符的batch_size都弄成了4了，先按照这个做下去。。
        # 为了醒目，此处空一行。。
        with tf.device('/cpu:0'):
            global_step = tf.Variable(40000, trainable=False)
            # 注：这个全局步数在测试的时候其实没有用，但是调用MaskMRCNN_model的时候需要一个参数，所以写进去。
            # 其实想想，就算这个全局步数需要用，这儿先给他个初始化的40000也没问题，因为恢复节点文件的时候，会把它改掉的（训练的时候就把它改掉了）。
        with tf.variable_scope("mask_rcnn_model"):
            mrcnn_feature_maps, detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, mrcnn_mask_mp, \
            rpn_rois, rpn_class, rpn_bbox, rpn_scores_nms, mrcnn_boxes_and_scores = \
                MaskRCNN_1_model.MaskMRCNN_model('inference', config, global_step, anchors, input_image_placeholder,
                                                 input_image_meta_placeholder,
                                                 input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                                 input_gt_class_ids_placeholder,
                                                 input_gt_boxes_placeholder, input_gt_masks_placeholder, input_sparse_gt, train_flag)
        all_vars = tf.all_variables()  # 应该是程序中所有变量。含有MRCNN的变量、GAN的变量、全局步数。
        MRCNN_vars = [var for var in all_vars if 'mask_rcnn_model' in var.name]  # MRCNN的变量。
        # 其实现在测试模式，可以用MRCNN_vars也可以用all_vars
        print("MRCNN的变量名（一共%d个）：" % len(MRCNN_vars))
        print(MRCNN_vars)
        sess = tf.Session()
        checkpoint_path_for_testing = './tmp/MRCNN-Resnet101-mp-' + mode_dataset + '-LISTA' + '-40000'
        path_for_testing = pathlib.Path(checkpoint_path_for_testing + '.meta')
        assert path_for_testing.is_file(), "没有训练节点问题，没法测试。"
        print("在以下路径发现了ckpt文件：", checkpoint_path_for_testing)
        checkpoints = cp.list_variables(checkpoint_path_for_testing)  # 注意这儿要用checkpoint_path而不是path。
        print("节点文件中的变量名（一共%d个）：" % len(checkpoints))
        print(checkpoints)  # 这儿打印出来的是节点文件中的变量名
        saver_MRCNN = tf.train.Saver(MRCNN_vars)
        # 测试的时候，没有那个GAN了，所以只恢复MRCNN的变量。不过，节点文件里还有GAN的变量，原来有点担心不对，结果
        #     测试发现居然可以恢复，看来节点文件里的变量多了，是没事儿的，只需要程序里有的，节点文件里都有，而且名字对得上就行。
        saver_MRCNN.restore(sess, checkpoint_path_for_testing)
        print("从本程序以前训练得到的节点文件中恢复了模型。")
        ################################################################################################################
        #                                                     下面，开始测试                                             #
        ################################################################################################################
        tests = int(len(dataset_test_ids)/config.BATCH_SIZE)  # 测试几次（相当于训练时候的max_steps）。每一次都是用config.BATCH_SIZE张图去做。
        co = 0  # 保存图像的名称，从几开始。。
        CPVs_all_test = np.zeros(
            [dataset_test_ids.shape[0], config.NUM_CLASSES - 1, config.NUM_CLASSES])  # 测试集、MP前的CPV（但是已经过了两轮选取）
        GTlabels_all_test = np.zeros([dataset_test_ids.shape[0], config.NUM_CLASSES - 1])  # 测试集、对应的金标准标签
        CPVs_all_test_MP = np.zeros(
            [dataset_test_ids.shape[0], config.NUM_CLASSES - 1, config.NUM_CLASSES])  # 测试集、MP后的CPV
        coord_error_all = []  # 这些是各种误差衡量标准
        AP_all = []
        classification_rate_all = []
        mean_iou_all = []
        precisions_all = []
        recalls_all = []
        APc_all = []
        coord_error_all_mp = []
        AP_all_mp = []
        classification_rate_all_mp = []
        mean_iou_all_mp = []
        precisions_all_mp = []
        recalls_all_mp = []
        APc_all_mp = []
        confuse_mat_err = []  # 识别有错，需要手动更新混淆矩阵的
        for step in range(tests):
            feed_dict, image_id_selected, real_patient_id, images, gt_class_ids, gt_boxs_pad, gt_masks = \
                feed_placeholder(dataset_test, config.BATCH_SIZE, [], anchors, config, num_rois, input_image_placeholder,
                                 input_image_meta_placeholder, input_rpn_match_placeholder, input_rpn_bbox_placeholder,
                                 input_gt_class_ids_placeholder, input_gt_boxes_placeholder, input_gt_masks_placeholder,
                                 input_gt_1_hot, input_sparse_gt, disp=True, is_training=False)  # dataset_train/dataset_test
            # 这种模式下input_labels_pl_grading其实是没用的，不过无所谓的，反正弄到feed_dict里不用也不会影响。
            molded_images, _, windows = MRCNN_utils.mold_inputs(config, images)
            # 以上是弄那个windows，keras程序里是同时输出了molded_images, image_metas，然后放到模型里去跑，
            #     但我感觉我应该不需要这样，因为我只需要保证测试时候用的图，和训练时候用的图，二者是用同一种方法搞出来的就可以了。
            #     这里留着molded_images是为了用它的shape（看了一下keras里，和原图大小是一样的）。
            _detections, _mrcnn_class, _mrcnn_bbox, _mrcnn_mask, _mrcnn_mask_mp, _rpn_rois, _rpn_scores_nms, \
            _rpn_class, _rpn_bbox, _mrcnn_boxes_and_scores = \
                sess.run([detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, mrcnn_mask_mp, rpn_rois, rpn_scores_nms,
                          rpn_class, rpn_bbox, mrcnn_boxes_and_scores], feed_dict=feed_dict)
            """
            beta=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mask_rcnn_model/mrcnn_class_bn1/beta:0")[0]
            gamma=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mask_rcnn_model/mrcnn_class_bn1/gamma:0")[0]
            m_mean=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mask_rcnn_model/mrcnn_class_bn1/moving_mean:0")[0]
            m_var=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mask_rcnn_model/mrcnn_class_bn1/moving_variance:0")[0]
            以上四句是想看看fpn_classifier_graph的第一个BN层，反正它的输入是x_bf1，输出是x_bf2，就看一下这些东西是怎么算出来的。
            可以用_beta, _gamma, _m_mean, _m_var = sess.run([beta, gamma, m_mean, m_var])
            然后发现，_beta和_gamma和输入的_x_bf1（一个批次的数据）的均值、方差完全不一样（shape是一样的，但是数不一样），也确实应该不一样，因为这俩应该是
                平均的平移和缩放参数，要是和这个批次的均值方差一样了，就相当于没有归一化了（见《185-基础1-BN.docx》里P2最下的东西）。
                后来测试了一下发现这个tf.contrib.layers.batch_norm确实是按照185中说的方法计算的归一化。见《15(BN_verify).py》。
            注意到，那个BN层是带ReLU的，所以，用《185-基础1-BN.docx》里归一化的公式验证了归一化之后，要ReLU一下才能得到_x_bf2。
            可以用以下两句把变量存出来：
            import scipy.io as sio
            sio.savemat('before_and_after_BN.mat', {'x_bf1': _x_bf1, 'x_bf2': _x_bf2, 'beta': _beta, 'gamma': _gamma, 'm_mean': _m_mean, 'm_var': _m_var})
            """
            ################################################################################################################
            #                                          下面，用没有信息传递的程序准备可视化                                     #
            ################################################################################################################
            results = []  #
            for i in range(config.BATCH_SIZE):
                final_rois, final_class_ids, final_scores, final_masks = \
                    unmold_MRCNN_outputs_for_visualization.unmold_detections \
                        (_detections[i], _mrcnn_mask[i], config.IMAGE_SHAPE, molded_images[i].shape, windows[i])
                """上句输入的时候，_mrcnn_mask[i]的shape是(100, 28, 28, 7)，这是说这张图里有100个提出，每个提出都是28*28的，然后有7种类别。
                看了一下_mrcnn_mask[i][0:,:,0]~[0:,:,6]，都是个28*28的矩阵，里面的数都是0~1之间的，但是加起来不等于1，说明存的应该是类似于logits的数。
                我现在想闹明白，他怎么就从这个玩意就变成掩膜了，掩膜怎么就变成logits了？？？
                    这是unmold_MRCNN_outputs_for_visualization.unmold_detections函数里做的：
                    首先是去掉了补零，就变成了(非零的提出个数, 28, 28, 7)，然后按照每个非零提出的类别，从该提出的7个掩膜里选1个，就成了(非零的提出个数, 28, 28)，
                    然后过滤掉面积为0的探测，最后把每个(28, 28)的掩膜放大到(512, 512)，并且根据外接矩形找到它在原图中的位置，就得到最后的输出final_masks，shape是(512, 512, 22)。

                【基础】可以在运行的时候把输入输出存到json文件里去，如下：
                    import json
                    file = open('unmold_detections_input.json', 'w', encoding='utf-8')
                    image_shape = list(image.shape)  # tuple变成list，否则读取的时候会出问题，JSONDecodeError: Expecting ',' delimiter: line一类的。。
                    molded_image_shape = list(molded_images[i].shape)
                    d = dict(detection=_detections[i].tolist(), mrcnn_mask=_mrcnn_mask[i].tolist(), 
                        image_shape=image_shape, molded_image_shape=molded_image_shape, window=windows[i].tolist())
                    json.dump(d, file, ensure_ascii=False)
                    file.close()
                    d1 = dict(final_rois=final_rois.tolist(), final_class_ids=final_class_ids.tolist(), 
                        final_scores=final_scores.tolist(), final_masks=final_masks.tolist())
                    file1 = open('unmold_detections_output.json', 'w', encoding='utf-8')
                    json.dump(d1, file1, ensure_ascii=False)
                    file1.close()
                    不过因为那个mrcnn_mask很大，所以存进去的时候有点慢。
                """
                results.append({"rois": final_rois, "class_ids": final_class_ids,
                                "scores": final_scores, "masks": final_masks, })
            """上句，把输出的探测结果编码成想要的形式。上面的结果只是用来和信息传递的结果作对比，并不是程序的最后结果。"""
            ################################################################################################################
            #                                          下面，用有信息传递的结果准备可视化                                      #
            ################################################################################################################
            results_mp = []  #
            """                                首先是准备工作，删除那些比较容易的错误，比如坐标值有错的。                       """
            _mrcnn_boxes_and_full_scores_1st_round, aver_score_1st_round, ix_selected_test_by_rpn_rois = \
                message_passing_manners.select_detections_for_mp_1st_round \
                    (config.NUM_CLASSES, config.RPN_MIN_CONFIDENCE_MP, _rpn_scores_nms,
                     _rpn_rois, _mrcnn_boxes_and_scores, _mrcnn_class)
            """                      以上，第一轮选取，删除掉rpn得分太低的mrcnn结果。当然，有免死金牌的就不删，见上。             """
            _mrcnn_boxes_and_full_scores_2nd_round = np.zeros_like(_mrcnn_boxes_and_full_scores_1st_round)
            # 第2轮选取结果，带其他类别分数的，初始化为0。
            IOU_threshold_one = config.MP_NMS_THRESHOLD_ONE
            # 阈值，如果mrcnn判断为背景类，且和某个前景类结果的IoU大于这个阈值，就认为mrcnn判断正确，确实是背景类，就把它删除。
            IOU_threshold_two = config.MP_NMS_THRESHOLD_TWO
            # 如果mrcnn判断为背景类，且和某两个前景类结果的IoU都大于这个阈值，也认为确实是背景类。
            for i in range(config.BATCH_SIZE):
                _mrcnn_boxes_and_full_scores_2nd_round_this, keep_iou1 = message_passing_manners.select_detections_for_mp_2nd_round \
                    (_mrcnn_boxes_and_full_scores_1st_round[i, :, :], aver_score_1st_round[i, :],
                     IOU_threshold_one, IOU_threshold_two, class_id_min, class_id_max)
                _mrcnn_boxes_and_full_scores_2nd_round[i, :, :] = _mrcnn_boxes_and_full_scores_2nd_round_this
                # 因为刚才是补的-1，所以mrcnn_boxes_and_scores_2nd_round后面补-1的地方，就全都是mrcnn_boxes_and_scores_1st_round的
                #     最后一行，所以就全都是0了。。
                ix_selected_end_round = ix_selected_test_by_rpn_rois[i, keep_iou1]
                # 上句，相对于输入进来的_mrcnn_boxes_and_scores_full，入选的几个索引号，后面选掩膜用的。这个不要补-1。
                _y_pred_test = _mrcnn_boxes_and_full_scores_2nd_round[:, :, 6:]  # 测试集的预测CPV。
                _y_pred_test_nz, nz = message_passing_manners.trim_zero(_y_pred_test[i, :, :])  # 非补零的CPV，还有非补零位置
                _boxes_pred = _mrcnn_boxes_and_full_scores_2nd_round[:, :, :4]  # 测试集的预测外接矩形。
                _boxes_pred_test_nz, _ = message_passing_manners.trim_zero(_boxes_pred[i, :, :])  # 非补零的外接矩形
                boxes_and_scores_this, _ = message_passing_manners.trim_zero(
                    _mrcnn_boxes_and_full_scores_2nd_round_this[:, :6])
                # 上句，第2轮选取后，外接矩形和得分（各类别的得分，现在就没用了）。
                l = _boxes_pred_test_nz.shape[0]  # 这儿是准备用SG滤波去除离群点了。
                x1_test = _boxes_pred_test_nz[:, 1]
                if l % 2 == 1:
                    window_length = l
                else:
                    window_length = l + 1
                x1_test_f = savgol_filter(x1_test, window_length=window_length, polyorder=2, mode='mirror')
                diff = abs(x1_test_f - x1_test)
                print('测试集，第%d张图，滤波误差最大是%.5f' % (real_patient_id[i], np.max(diff)))
                outlier = np.where(diff > _SG_thre)[0]
                if outlier.shape[0] != 0:
                    _y_pred_test_nz = np.delete(_y_pred_test_nz, outlier, 0)  # 【基础】最后的0是表示删除第outlier这一整行
                    boxes_and_scores_this = np.delete(boxes_and_scores_this, outlier, 0)
                    print('删除了第', outlier, '个。')
                else:
                    pass
                """           以上，第二轮选取、滤波去除离群点，其实，如果用那个SG滤波的话，也许这一轮就不用了，但没试过。          """
                # CPVs_all_test[2*step+i, :_y_pred_test_nz.shape[0], :] = _y_pred_test_nz  # 把这张图的、两轮选取后修正前的、去补零的CPV，保存下来。但是，要注意到这个结果其实还是有补零的，因为这个CPVs_all初始化为全零的，现在只是把原来非零的地方给覆盖掉了。
                # GTlabels_all_test[2*step+i, :_y_pred_test_nz.shape[0]] = gt_class_ids[i, :_y_pred_test_nz.shape[0]][::-1]
                # 上句，是把对应的金标准弄出来，因为原来金标准是按照y坐标降序排列的，所以是1 2 3 4...8这样的升序，现在既然预测结果是
                #     按照y坐标升序排列的，所以应该倒序一下，按照降序排列吧。。
                """           以上，保存文件，都是为了调试那个信息传递用的（准备存成mat文件），对于程序执行本身用处不算大。。       """
                y_pred_test_nz = tf.convert_to_tensor(_y_pred_test_nz, dtype=tf.float32)
                # 上句，因为那个信息传递是用tf的，所以把这个东西给变成张量，
                y_pred_test_temp = tf.expand_dims(y_pred_test_nz, axis=0)
                # 上句，变成(1, ?, config.NUM_CLASSES)的。
                # 【基础】现在这个虽然是?，但是由于是从np的东西变过来的，所以对于每张图，这个?就是已知的了，比如说第0张图有8个节点，那么这个?就是8，
                #     然后弄过来的y_pred_test_temp就是(1, 8, 10)的，下一个可能是(1, 7, 10)的，这个没关系，都可以用tf的函数了。
                # 也就是说，只要他到tf函数之前是知道大小（而不是用?代替的大小）的，那么就可以。
                y_test_calibrated_temp = message_passing_manners.get_calibrated_CPV(y_pred_test_temp, Phi_mat, Psi_mat)
                # 上句，用信息传递修正标签，得到的这一张图的标签
                y_test_calibrated = tf.squeeze(y_test_calibrated_temp)  # 变回(config.NUM_CLASSES-1, config.NUM_CLASSES)的
                labels_calibrated = tf.argmax(y_test_calibrated, axis=1)
                _y_test_calibrated, _labels_calibrated = sess.run([y_test_calibrated, labels_calibrated],
                                                                  feed_dict=feed_dict)
                # 上面，再给他给run回来。
                """                 以上，信息传递的修正CPV。先变成张量，用那个tf的信息传递函数处理之后，再run回来。             """
                # CPVs_all_test_MP[2*step+i, :_y_test_calibrated.shape[0], :] = _y_test_calibrated  # 把这张图的、信息传递后的CPV，保存下来（准备存成mat文件）。
                ################################################################################################################
                #                              下面，为了增强Wt（_Psi_mat）的作用，试试用viterbi算法处理一下                         #
                ################################################################################################################
                nodes_test = [dict(zip(['0BG', '0S1', '0L5', '0L4', '0L3', '0L2', '0L1', 'T12', 'T11', 'T10'], i))
                              for i in _y_test_calibrated]
                # '0BG', '0S1', '0L5', '0L4', '0L3', '0L2', '0L1', 'T12', 'T11', 'T10'
                # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
                tags_test = message_passing_manners.viterbi(nodes_test, trans)[0]  # viterbi增强转移矩阵的作用
                print('经过CRF修正后的标签：', tags_test)
                s1 = map(''.join, zip(*[iter(tags_test)] * 3))
                labels = list(s1)
                _y_test_calibrated1 = [letterdict[label] for label in labels]
                _y_test_calibrated1 = np.array(_y_test_calibrated1)
                """                                          以上，viterbi算法处理完毕。                                   """
                ################################################################################################################
                #                            下面是收尾工作，把修正的概率预测值对应回相应的外接矩形和掩膜上去。                        #
                ################################################################################################################
                _detections_corrected = boxes_and_scores_this
                _detections_corrected[:, 4] = _y_test_calibrated1
                # 以上两句，是经过信息传递修正标签后的探测结果。如果用_y_test_calibrated1，就是有viterbi的，否则是没有的。
                detection_boxes = _detections_corrected[:, :4]  # 这个是要把探测结果（归一化的外接矩形）弄到0~1之间。
                detection_boxes[detection_boxes < 0] = 0.0
                detection_boxes[detection_boxes > 1] = 1.0
                _detections_corrected[:, :4] = detection_boxes  # 弄完了的放回去。
                _detections_corrected_mrcnn_mask = _mrcnn_mask_mp[i, ix_selected_end_round, :, :, :]
                # 上句，把掩膜重新排序。这是因为，信息传递对检测的外接矩形重新排序了，所以也得把对应的掩膜排好序。
                """
                【注意】
                一个挺让人晕乎的地方：掩膜和探测结果的对应关系
                原来不要信息传递的时候，在MaskRCNN_1_model里把detection_boxes输入到了build_fpn_mask_graph_with_reuse函数里，
                    所以它输出的掩膜（即mrcnn_mask）的顺序，是按照detection_boxes的顺序排的，而这个detection_boxes里的几个外接矩形
                    既不是按照坐标，也不是按照信心分值排序的，不知道他怎么弄的，反正挺烦的，但当时只要外接矩形和掩膜是对应的，也就无所谓了；
                现在要信息传递的时候，我其实就可以不用他的detection_boxes了，而直接用mrcnn_class、mrcnn_bbox，也就是类别、信心、外接矩形，
                    这些信息都包含在mrcnn_boxes_and_scores里。然后，把mrcnn_bbox修正过的rpn_rois（记作mrcnn_boxes）输入到
                    build_fpn_mask_graph_with_reuse函数里，就得到了按照mrcnn_boxes_and_scores的顺序排列的掩膜了（即mrcnn_mask_mp）。
                然后，在信息传递里对mrcnn_boxes_and_scores的顺序进行了重排，所以要记录下来序号（这儿选了两次，最终是得到ix_selected_end_round），
                    用这个ix_selected_end_round把掩膜mrcnn_mask_mp也重排了，然后再做后面的unmold_detections函数，就可以了。
                还有一个有意思的事儿，我发现把mrcnn_boxes_and_scores输入到build_fpn_mask_graph_with_reuse函数里，
                得到的掩膜比把detection_boxes输入进去的时候要好。
                """
                final_rois_mp, final_class_ids_mp, final_scores_mp, final_masks_mp = \
                    unmold_MRCNN_outputs_for_visualization.unmold_detections \
                        (_detections_corrected, _detections_corrected_mrcnn_mask, config.IMAGE_SHAPE,
                         molded_images[i].shape, windows[i])
                results_mp.append({"rois": final_rois_mp, "class_ids": final_class_ids_mp,
                                   "scores": final_scores_mp, "masks": final_masks_mp})
                """                                          以上，可视化准备工作完毕。                                    """
            ################################################################################################################
            #                                          以下，可视化、计算误差、AP值、保存。                                    #
            ################################################################################################################
            for j in range(config.BATCH_SIZE):  # 可视化
                savename = './result/%d-gt.png' % co
                original_image = images[j, :, :, :]
                gt_class_id = gt_class_ids[j, :]
                nz = np.where(gt_class_id != 0)  # 找到不为0的地方（去掉补零），这儿得的是一个tuple
                nz = nz[0]  # tuple变成np.array
                gt_class_id = gt_class_id[nz]
                gt_box = gt_boxs_pad[j, :, :]
                gt_box = gt_box[nz]  # 去掉补零的外接矩形。注意到，那个_mrcnn_bbox是修正值，虽然都叫bbox，但是所指不同。
                gt_mask = gt_masks[j, :, :, :]
                gt_mask = gt_mask[:, :, nz]
                gt_mask = np.reshape(gt_mask, [gt_mask.shape[0], gt_mask.shape[1], nz.shape[0]])
                gt_mask_back = MRCNN_utils.expand_mask(gt_box, gt_mask, original_image.shape[:2])
                visualize.display_instances(original_image, gt_box, gt_mask_back,
                                            gt_class_id, dataset_test.class_names, savename, figsize=(8, 8))
                # 那个小掩膜的事儿，，是不是得加个if啊，如果没有小掩膜，岂不是悲剧的？？？
                r = results[j]
                r_mp = results_mp[j]
                savename = './result/%d-results.png' % co
                savename_mp = './result/%d-results-mp.png' % co
                visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                            dataset_test.class_names, savename, r['scores'], figsize=(8, 8))
                visualize.display_instances(original_image, r_mp['rois'], r_mp['masks'], r_mp['class_ids'],
                                            dataset_test.class_names, savename_mp, r_mp['scores'], figsize=(8, 8))
                #  等等试试。。
                # savename1 = '%d-results_only_boxes.png' % co
                # full_mask_zhangdong = visualize.draw_full_mask(r['class_ids'], r['masks'])  # 给张栋的全掩膜图
                # full_mask_zhangdong_gt = visualize.draw_full_mask(gt_class_id, gt_mask_back)  # 给张栋的全掩膜图金标准
                # import matplotlib.pyplot as plt  # 给张栋的全掩膜图
                # plt.figure(1)  # 给张栋的全掩膜图
                # plt.imshow(full_mask_zhangdong)  # 给张栋的全掩膜图
                # plt.savefig('full_mask_zhangdong_%d.png' % co, bbox_inches="tight", pad_inches=0, ax=8)  # 给张栋的全掩膜图
                # plt.imshow(full_mask_zhangdong_gt)  # 给张栋的全掩膜图
                # plt.savefig('full_mask_zhangdong_gt_%d.png' % co, bbox_inches="tight", pad_inches=0, ax=8)  # 给张栋的全掩膜图
                # sio.savemat('zhangdong_%d.mat' % co, {'pred': full_mask_zhangdong, 'gt': full_mask_zhangdong_gt})  # 给张栋的全掩膜图
                # try:
                #     visualize.draw_detection_and_gt(original_image, savename1, boxes=r_mp['rois'][::-1], gt=gt_box,
                #                                     captions=None, class_ids=r_mp['class_ids'][::-1],
                #                                     class_names=dataset_test.class_names, scores=r_mp['scores'][::-1])
                #     # [::-1]是为了让结果倒序排列，好和金标准的顺序对应上。
                # except:
                #     print('检查')
                AP, precisions, recalls, overlaps_all, classification_rate, mean_iou, APc = \
                    MRCNN_utils.compute_ap(config.NUM_CLASSES, gt_box, gt_class_id, r["rois"], r["class_ids"],
                                           r["scores"], iou_threshold=0.75)
                # 上句，默认是用IoU阈值为0.5的时候计算的各个指标，改变阈值，可以得到不同阈值时候的指标。
                print('验证结果：AP值%.3f，分类准确度%.3f，分类正确的提出的重合度%.3f' % (
                AP, classification_rate, mean_iou))  # 分类准确度，是搞对的除以预测总数（而不是金标准总数）
                print('准确率曲线：')
                print(precisions)
                print('召回率曲线:')
                print(recalls)
                coord_error_this = MRCNN_utils.compute_coord_error(gt_box, r["rois"])
                coord_error_all.append(coord_error_this)
                AP_all.append(AP)
                APc_all.append(np.array(APc))  # 这个在这儿其实是没用的，滑脱的时候有用的。
                classification_rate_all.append(classification_rate)
                mean_iou_all.append(mean_iou)
                precisions_all.append(precisions)
                recalls_all.append(recalls)
                AP_mp, precisions_mp, recalls_mp, overlaps_all_mp, classification_rate_mp, mean_iou_mp, APc_mp = \
                    MRCNN_utils.compute_ap(config.NUM_CLASSES, gt_box, gt_class_id, r_mp["rois"], r_mp["class_ids"],
                                           r_mp["scores"], iou_threshold=0.75)
                # 上句，默认是用IoU阈值为0.5的时候计算的各个指标，改变阈值，可以得到不同阈值时候的指标。
                print('信息传递后，验证结果：AP值%.3f，分类准确度%.3f，分类正确的提出的重合度%.3f' % (AP_mp, classification_rate_mp, mean_iou_mp))
                print('信息传递后，准确率曲线：')
                print(precisions_mp)
                print('信息传递后，召回率曲线:')
                print(recalls_mp)
                coord_error_this_mp = MRCNN_utils.compute_coord_error(gt_box, r_mp["rois"])
                coord_error_all_mp.append(coord_error_this_mp)
                AP_all_mp.append(AP_mp)
                APc_all_mp.append(np.array(APc_mp))
                classification_rate_all_mp.append(classification_rate_mp)
                mean_iou_all_mp.append(mean_iou_mp)
                precisions_all_mp.append(precisions_mp)
                recalls_all_mp.append(recalls_mp)
                if len(gt_class_id) == len(r_mp['class_ids'][::-1]):
                    if np.sum(gt_class_id - r_mp['class_ids'][::-1]) == 0:
                        print('分类正确')
                    else:
                        print('警告：第%d张，真实索引号%d，分类错误，需要手动更新癌症混淆矩阵。' % (image_id_selected[j], real_patient_id[j]))
                        confuse_mat_err.append(image_id_selected[j])
                else:
                    print('警告：第%d张，真实索引号%d，漏检或多检，需要手动更新癌症混淆矩阵。' % (image_id_selected[j], real_patient_id[j]))
                    confuse_mat_err.append(image_id_selected[j])
                co = co + 1
        # sa = 'CPV_test_before_MP.mat'  #
        # sio.savemat(sa, {'CPVs': CPVs_all_test})  # 存一下测试集中、输入信息传递的东西。
        # CPV_test_before_MP_and_gt_label = np.concatenate([CPVs_all_test, np.expand_dims(GTlabels_all_test, axis=2)], axis=2)
        # sa2 =  'CPV_test_before_MP_with_gt_lab.mat'  #
        # sio.savemat(sa2, {'CPVs': CPV_test_before_MP_and_gt_label})
        # sa1 = 'CPV_test_MP.mat'  #
        # sio.savemat(sa1, {'CPVs': CPVs_all_test_MP})  # 存一下测试集中、信息传递输出的东西。
        coord_error_all = np.array(coord_error_all)
        print('像素点平均误差（不带信息传递的）是：', np.mean(coord_error_all))  # 17.960(split5)，因为有没检测出来的补了0，用0去减就很大啊。稍微减小一些吧。。
        AP_all = np.array(AP_all)  # 测试集中所有样本的AP值，这个AP的定义类似于《226说明1-average precision》中所有召回率对应的PR曲线下面积。
        APc_all = np.array(APc_all)
        classification_rate_all = np.array(classification_rate_all)
        mean_iou_all = np.array(mean_iou_all)
        precisions_all = np.array(precisions_all)  #
        recalls_all = np.array(recalls_all)
        savename2 = 'validations_dataset_test_' + mode_dataset + '_0.75.mat'  # validations_dataset_train/validations_dataset_test
        sio.savemat(savename2,
                    {'AP_all': AP_all, 'APc_all': APc_all, 'classification_rate_all': classification_rate_all,
                     'mean_iou_all': mean_iou_all, 'precisions_all': precisions_all, 'recalls_all': recalls_all})
        coord_error_all_mp = np.array(coord_error_all_mp)
        print('像素点平均误差（带信息传递的）是：', np.mean(coord_error_all_mp))  # 2.699(split5)
        AP_all_mp = np.array(AP_all_mp)  # 测试集中所有样本的AP值，这个AP的定义类似于《226说明1-average precision》中所有召回率对应的PR曲线下面积。
        APc_all_mp = np.array(APc_all_mp)
        classification_rate_all_mp = np.array(classification_rate_all_mp)
        mean_iou_all_mp = np.array(mean_iou_all_mp)
        precisions_all_mp = np.array(precisions_all_mp)  #
        recalls_all_mp = np.array(recalls_all_mp)
        savename3 = 'validations_dataset_test_' + mode_dataset + '_0.75_mp.mat'  # validations_dataset_train/validations_dataset_test
        sio.savemat(savename3,
                    {'AP_all': AP_all_mp, 'APc_all': APc_all_mp, 'classification_rate_all': classification_rate_all_mp,
                     'mean_iou_all': mean_iou_all_mp, 'precisions_all': precisions_all_mp,
                     'recalls_all': recalls_all_mp})
        print('需要手动更新混淆矩阵的几个：', confuse_mat_err)
        print('测试结束。')


if __name__ == '__main__':
    tf.app.run()