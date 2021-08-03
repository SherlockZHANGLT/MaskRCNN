import tensorflow as tf
import numpy as np
import scipy.io as sio
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

def delete_incorrect_L4_and_S1(final_rois, final_class_ids, final_scores, final_masks):
    """
    180925，大幅化简这个程序。因为现在通过加GAN、规定target_class_ids数量、正负例个数等措施，已经可以把测试阶段的提出弄得还好了。
    所以，为何不在这三种情况下，直接根据位置、分数去做投票，然后直接选择得票最高的呢？！

    思路是，在测试的时候把那个DETECTION_MIN_CONFIDENCE改得更小一些，保证他不会丢掉正确的探测物体。
        当然这样的结果就会多引入一些错误的探测结果。但没关系，因为多弄出来的东西，信心分值肯定很小（也就是0.6啊0.7啊什么的），
        对比那些0.98以上的，很容易就删除掉。
    所以，也不用去看有多少个L4、多少个L5、多少个S1了，就把每一类的所有的东西都找出来，

    本函数的输入：
        （注意输入的第0维度不是批次大小，而是这张图中的检测到的物体数）
        final_rois：(3,4)，也有可能是(4,4)
        final_class_ids：(3,)，也有可能是(4,)
        final_scores： (3,)，也有可能是(4,)
        final_masks：(512, 512, 3)，也有可能是(512, 512, 4)
    """
    L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 那些被认为是L4的探测结果，对应于final_rois/final_class_ids/...里的第几行
    # 例如，如果第2个探测结果是L4，那么这个就是array([2], dtype=int64)，如果第0和2个探测结果都是L4（相当于多了一个L4），那么这个就是
    #     array([0, 2], dtype=int64)。
    L5_pos = np.array(np.where(final_class_ids == 2))[0]
    S1_pos = np.array(np.where(final_class_ids == 3))[0]  # array([1, 3], dtype=int64) 这个地方各种加[0]什么的特别烦
    # assert L4_pos.shape[0] > 0, '没检测到L4，需要降低config.DETECTION_MIN_CONFIDENCE重新执行测试程序。'
    # assert L5_pos.shape[0] > 0, '没检测到L5，需要降低config.DETECTION_MIN_CONFIDENCE重新执行测试程序。'
    # assert S1_pos.shape[0] > 0, '没检测到S1，需要降低config.DETECTION_MIN_CONFIDENCE重新执行测试程序。'
    if L4_pos.shape[0] > 0 and L5_pos.shape[0] > 0 and S1_pos.shape[0] > 0:
        L4_boxes = final_rois[L4_pos, :]  # 那些被认为是L4的探测结果的外接矩形
        L5_boxes = final_rois[L5_pos, :]  # 那些被认为是L5的探测结果的外接矩形
        S1_boxes = final_rois[S1_pos, :]  # 那些被认为是S1的探测结果的外接矩形
        L4_scores = final_scores[L4_pos]  # 那些被认为是L4的探测结果的信心分值
        L5_scores = final_scores[L5_pos]  # 那些被认为是L5的探测结果的信心分值
        S1_scores = final_scores[S1_pos]  # 那些被认为是S1的探测结果的信心分值
        L4_masks = final_masks[:, :, L4_pos]  # 那些被认为是L4的探测结果的掩膜
        L5_masks = final_masks[:, :, L5_pos]  # 那些被认为是L5的探测结果的掩膜
        S1_masks = final_masks[:, :, S1_pos]  # 那些被认为是S1的探测结果的掩膜
        L4_scores_selection = np.zeros_like(L4_pos, dtype=np.float32)  # 那些被认为是L4的探测结果的投票结果，初始化为0票。
        L5_scores_selection = np.zeros_like(L5_pos, dtype=np.float32)
        S1_scores_selection = np.zeros_like(S1_pos, dtype=np.float32)
        """以上是准备，以下是处理L4的情况。先用信心分值投票，然后再看如果只有1个L5，就利用坐标信息。"""
        if L4_scores.shape[0]>1:  # 处理L4的情况，如果有多个，就投票法；如果就一个，就选了。
            L4_scores_sorted = np.sort(L4_scores)  # 所有认为是L4的东西，按照信心分值从小到大排列。
            L4_scores_sorted = L4_scores_sorted[::-1]  # 倒序过来，从大到小。
            L4_scores_sorted_pos = np.argsort(L4_scores)  # 所有认为是L4的东西，按照信心分值从小到大排列后的顺序.
            L4_scores_sorted_pos = L4_scores_sorted_pos[::-1]  # 倒序过来，从大到小。
            for i in range(L4_scores.shape[0]):
                if i == 0:
                    pass  # 信心分值最大的，保留为0分
                else:
                    score_diff = L4_scores_sorted[0] - L4_scores_sorted[i]  # 这一个被认为是L4的，分值比最高的低了多少。
                    L4_scores_selection[i] = L4_scores_selection[i] - score_diff  # 扣除这个分值。
            # 上面for循环，相当于是先按照信心分值排序。如果下面所有if条件都不满足，那么就按照信心分值选择了；
            #     而，这几个被认为是L4的东西，信心分值往往相差很少，也就是0.0几甚至0.00几。如果满足下面的强条件，那么这个自然可以忽略不计。
            if L4_scores_sorted[0] > L4_scores_sorted[1] + 0.2:  # 如果信心分值最大的，比信心分值第二大的大了0.2以上，说明信心很充分的。
                L4_scores_selection[L4_scores_sorted_pos[0]] = L4_scores_selection[L4_scores_sorted_pos[0]] + 3  # 直接投3票
            elif L4_scores_sorted[0] > L4_scores_sorted[1] + 0.1:  # 如果信心分值最大的，比信心分值第二大的大了0.1以上，说明信心也挺充分的。
                L4_scores_selection[L4_scores_sorted_pos[0]] = L4_scores_selection[L4_scores_sorted_pos[0]] + 2  # 直接投2票
            elif L4_scores_sorted[0] > L4_scores_sorted[1] + 0.05:  # 如果信心分值最大的，比信心分值第二大的大了0.05以上，说明还有信心。
                L4_scores_selection[L4_scores_sorted_pos[0]] = L4_scores_selection[L4_scores_sorted_pos[0]] + 1  # 投1票
            elif L4_scores_sorted[0] > 0.99:
                low_score = np.where(L4_scores_sorted<=L4_scores_sorted[0]-0.05)[0]
                L4_scores_selection[low_score] = L4_scores_selection[low_score] - 0.5  # 分数有点低（差了0.05），那么投-0.5票
                too_low_score = np.where(L4_scores_sorted <= L4_scores_sorted[0] - 0.1)[0]
                L4_scores_selection[too_low_score] = L4_scores_selection[too_low_score] - 0.5  # 分数特别低（差了0.1），那么再投-0.5票，一共-1票。
            else:
                print('最大的两个L4信心分值相差不到0.05，不投票。')
            if L5_pos.shape[0] == 1:
                """以下只对只有1个L5的情况有效。"""
                mean_dist = np.mean(abs(L5_boxes - L4_boxes), axis=1)  # 所有被认为是L4的东西，和那个L5的外接矩形的四个角点的平均绝对距离
                # mean_dist_sorted = np.sort(mean_dist)  # 平均距离从小到大排序
                mean_dist_sorted_pos = np.argsort(mean_dist)  # 平均距离从小到大排序，对应的位置索引。
                # 这个不要倒序，就是从小到大排列，然后距离最小的那个（即离L5最近的那个L4），对应的位置投1票。
                L4_scores_selection[mean_dist_sorted_pos[0]] = L4_scores_selection[mean_dist_sorted_pos[0]] + 0.5  # 投0.5票
                """以上是找和L5的平均距离。"""
                x1_L4 = L4_boxes[:, 1]
                x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
                x1_dist_lt_50_pos = np.where(abs(x1_L4-x1_L5) < 50)  # 找到横向距离小于这个阈值50的那些探测结果
                L4_scores_selection[x1_dist_lt_50_pos] = L4_scores_selection[x1_dist_lt_50_pos] + 0.5  # 投0.5票
                x2_L4 = L4_boxes[:, 3]
                x2_L5 = L5_boxes[:, 3][0]  # 加一个[0]把1维的array变成数
                x2_dist_lt_50_pos = np.where(abs(x2_L4 - x2_L5) < 50)  # 找到横向距离小于这个阈值50的那些探测结果
                L4_scores_selection[x2_dist_lt_50_pos] = L4_scores_selection[x2_dist_lt_50_pos] + 0.5  # 投0.5票
                """以上是检查和L5的横向距离，因为三块椎骨的横向距离不应该差太远的。"""
            L4_area_normed = (L4_boxes[:, 2] - L4_boxes[:, 0]) * (L4_boxes[:, 3] - L4_boxes[:, 1])/512/512  # 512日后改成h/w
            too_small = (L4_area_normed<0.001)  # 一般而言，椎骨的面积是在总面积的0.01左右，所以，小于10倍或者大于20倍就认为他不对。
            too_large = (L4_area_normed>0.2)
            somewhat_small = (L4_area_normed < 0.002)  # 后来加上，小于1/5或者大于5倍，就是小于0.002或者大于0.05，也是有点不对的，不减2票，但是可以减1票。
            somewhat_large = (L4_area_normed > 0.05)
            incorrect_by_area = np.where(np.logical_or(too_small, too_large))
            maybe_incorrect_by_area = np.where(np.logical_or(somewhat_small, somewhat_large))
            L4_scores_selection[maybe_incorrect_by_area] = L4_scores_selection[maybe_incorrect_by_area] - 1  # 减1票（可能不对的）
            L4_scores_selection[incorrect_by_area] = L4_scores_selection[incorrect_by_area] - 1  # 再减1票，一共减2票（因为上面靠距离加的票，最多是1.5票）
            L4_scores_selection_sorted_pos = np.argsort(L4_scores_selection)  # 投票结果从小到大排列的顺序
            L4_selected = L4_scores_selection_sorted_pos[-1]  # 选最后一个，不确定行不行呢。。。
            L4_rois_selected = L4_boxes[L4_selected]
            L4_scores_selected = L4_scores[L4_selected]
            L4_masks_selected = L4_masks[:, :, L4_selected]
            L4_masks_selected = np.expand_dims(L4_masks_selected, axis=2)
        else:
            L4_rois_selected = L4_boxes[0]  # 这地方得加0，否则shape不一样。
            L4_scores_selected = L4_scores[0]
            L4_masks_selected = L4_masks
        """以上处理完L4，以下处理S1，方法类似。"""
        if S1_scores.shape[0]>1:  # 处理S1的情况，如果有多个，就投票法；如果就一个，就选了。
            S1_scores_sorted = np.sort(S1_scores)  # 所有认为是S1的东西，按照信心分值从小到大排列。
            S1_scores_sorted = S1_scores_sorted[::-1]  # 倒序过来，从大到小。
            S1_scores_sorted_pos = np.argsort(S1_scores)  # 所有认为是S1的东西，按照信心分值从小到大排列后的顺序.
            S1_scores_sorted_pos = S1_scores_sorted_pos[::-1]  # 倒序过来，从大到小。
            for i in range(S1_scores.shape[0]):
                if i == 0:
                    pass  # 信心分值最大的，保留为0分
                else:
                    score_diff = S1_scores_sorted[0] - S1_scores_sorted[i]  # 这一个被认为是S1的，分值比最高的低了多少。
                    S1_scores_selection[i] = S1_scores_selection[i] - score_diff  # 扣除这个分值。
            # 上面for循环，相当于是先按照信心分值排序。如果下面所有if条件都不满足，那么就按照信心分值选择了；
            #     而，这几个被认为是S1的东西，信心分值往往相差很少，也就是0.0几甚至0.00几。如果满足下面的强条件，那么这个自然可以忽略不计。
            if S1_scores_sorted[0] > S1_scores_sorted[1] + 0.2:  # 如果信心分值最大的，比信心分值第二大的大了0.2以上，说明信心很充分的。
                S1_scores_selection[S1_scores_sorted_pos[0]] = S1_scores_selection[S1_scores_sorted_pos[0]] + 3  # 直接投3票
            elif S1_scores_sorted[0] > S1_scores_sorted[1] + 0.1:  # 如果信心分值最大的，比信心分值第二大的大了0.1以上，说明信心也挺充分的。
                S1_scores_selection[S1_scores_sorted_pos[0]] = S1_scores_selection[S1_scores_sorted_pos[0]] + 2  # 直接投2票
            elif S1_scores_sorted[0] > S1_scores_sorted[1] + 0.05:  # 如果信心分值最大的，比信心分值第二大的大了0.05以上，说明还有信心。
                S1_scores_selection[S1_scores_sorted_pos[0]] = S1_scores_selection[S1_scores_sorted_pos[0]] + 1  # 投1票
            elif S1_scores_sorted[0] > 0.99:
                low_score = np.where(S1_scores_sorted<=S1_scores_sorted[0]-0.05)[0]
                S1_scores_selection[low_score] = S1_scores_selection[low_score] - 0.5
                too_low_score = np.where(S1_scores_sorted <= S1_scores_sorted[0] - 0.1)[0]
                S1_scores_selection[too_low_score] = S1_scores_selection[too_low_score] - 0.5
            else:
                print('最大的两个S1信心分值相差不到0.05，不投票。')
            if L5_pos.shape[0] == 1:
                """以下只对只有1个L5的情况有效。"""
                mean_dist = np.mean(abs(S1_boxes - L5_boxes), axis=1)  # 所有被认为是S1的东西，和那个L5的外接矩形的四个角点的平均绝对距离
                # mean_dist_sorted = np.sort(mean_dist)  # 平均距离从小到大排序
                mean_dist_sorted_pos = np.argsort(mean_dist)  # 平均距离从小到大排序，对应的位置索引。
                # 这个不要倒序，就是从小到大排列，然后距离最小的那个（即离L5最近的那个L4），对应的位置投1票。
                S1_scores_selection[mean_dist_sorted_pos[0]] = S1_scores_selection[mean_dist_sorted_pos[0]] + 0.5  # 投0.5票
                """以上是找和L5的平均距离。"""
                x1_S1 = S1_boxes[:, 1]
                x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
                x1_dist_lt_50_pos = np.where(abs(x1_S1 - x1_L5) < 50)  # 找到横向距离小于这个阈值50的那些探测结果
                S1_scores_selection[x1_dist_lt_50_pos] = S1_scores_selection[x1_dist_lt_50_pos] + 0.5  # 投0.5票
                x2_S1 = S1_boxes[:, 3]
                x2_L5 = L5_boxes[:, 3][0]  # 加一个[0]把1维的array变成数
                x2_dist_lt_50_pos = np.where(abs(x2_S1 - x2_L5) < 50)  # 找到横向距离小于这个阈值50的那些探测结果
                S1_scores_selection[x2_dist_lt_50_pos] = S1_scores_selection[x2_dist_lt_50_pos] + 0.5  # 投0.5票
                """以上是检查和L5的横向距离，因为三块椎骨的横向距离不应该差太远的。"""
            S1_area_normed = (S1_boxes[:, 2] - S1_boxes[:, 0]) * (S1_boxes[:, 3] - S1_boxes[:, 1]) / 512 / 512  # 512日后改成h/w
            too_small = (S1_area_normed < 0.001)  # 一般而言，椎骨的面积是在总面积的0.01左右，所以，小于10倍或者大于20倍就认为他不对。
            too_large = (S1_area_normed > 0.2)
            somewhat_small = (S1_area_normed < 0.002)  # 后来加上，小于1/5或者大于5倍，就是小于0.002或者大于0.05，也是有点不对的，不减2票，但是可以减1票。
            somewhat_large = (S1_area_normed > 0.05)
            incorrect_by_area = np.where(np.logical_or(too_small, too_large))
            maybe_incorrect_by_area = np.where(np.logical_or(somewhat_small, somewhat_large))
            S1_scores_selection[maybe_incorrect_by_area] = S1_scores_selection[maybe_incorrect_by_area] - 1  # 减1票（可能不对的）
            S1_scores_selection[incorrect_by_area] = S1_scores_selection[incorrect_by_area] - 1  # 再减1票，一共减2票（因为上面靠距离加的票，最多是1.5票）
            S1_scores_selection_sorted_pos = np.argsort(S1_scores_selection)  # 投票结果从小到大排列的顺序
            S1_selected = S1_scores_selection_sorted_pos[-1]  # 选最后一个，不确定行不行呢。。。
            S1_rois_selected = S1_boxes[S1_selected]
            S1_scores_selected = S1_scores[S1_selected]
            S1_masks_selected = S1_masks[:, :, S1_selected]
            S1_masks_selected = np.expand_dims(S1_masks_selected, axis=2)
        else:
            S1_rois_selected = S1_boxes[0]
            S1_scores_selected = S1_scores[0]
            S1_masks_selected = S1_masks
        """以上处理完S1，以下处理L5，信心分值的方法类似，但是距离的事儿方法就不一样了。"""
        if L5_scores.shape[0]>1:  # 处理L5的情况，如果有多个，就投票法；如果就一个，就选了。
            L5_scores_sorted = np.sort(L5_scores)  # 所有认为是L5的东西，按照信心分值从小到大排列。
            L5_scores_sorted = L5_scores_sorted[::-1]  # 倒序过来，从大到小。
            L5_scores_sorted_pos = np.argsort(L5_scores)  # 所有认为是L5的东西，按照信心分值从小到大排列后的顺序.
            L5_scores_sorted_pos = L5_scores_sorted_pos[::-1]  # 倒序过来，从大到小。
            for i in range(L5_scores.shape[0]):
                if i == 0:
                    pass  # 信心分值最大的，保留为0分
                else:
                    score_diff = L5_scores_sorted[0] - L5_scores_sorted[i]  # 这一个被认为是L5的，分值比最高的低了多少。
                    L5_scores_selection[i] = L5_scores_selection[i] - score_diff  # 扣除这个分值。
            # 上面for循环，相当于是先按照信心分值排序。如果下面所有if条件都不满足，那么就按照信心分值选择了；
            #     而，这几个被认为是L5的东西，信心分值往往相差很少，也就是0.0几甚至0.00几。如果满足下面的强条件，那么这个自然可以忽略不计。
            if L5_scores_sorted[0] > L5_scores_sorted[1] + 0.2:  # 如果信心分值最大的，比信心分值第二大的大了0.2以上，说明信心很充分的。
                L5_scores_selection[L5_scores_sorted_pos[0]] = L5_scores_selection[L5_scores_sorted_pos[0]] + 3  # 直接投3票
            elif L5_scores_sorted[0] > L5_scores_sorted[1] + 0.1:  # 如果信心分值最大的，比信心分值第二大的大了0.1以上，说明信心也挺充分的。
                L5_scores_selection[L5_scores_sorted_pos[0]] = L5_scores_selection[L5_scores_sorted_pos[0]] + 2  # 直接投2票
            elif L5_scores_sorted[0] > L5_scores_sorted[1] + 0.05:  # 如果信心分值最大的，比信心分值第二大的大了0.05以上，说明还有信心。
                L5_scores_selection[L5_scores_sorted_pos[0]] = L5_scores_selection[L5_scores_sorted_pos[0]] + 1  # 投1票
            elif L5_scores_sorted[0] > 0.99:
                low_score = np.where(L5_scores_sorted<=L5_scores_sorted[0]-0.05)[0]
                L5_scores_selection[low_score] = L5_scores_selection[low_score] - 0.5
                too_low_score = np.where(L5_scores_sorted <= L5_scores_sorted[0] - 0.1)[0]
                L5_scores_selection[too_low_score] = L5_scores_selection[too_low_score] - 0.5
            else:
                print('最大的两个L5信心分值相差不到0.05，不投票。')
            if L4_pos.shape[0] == 1 and S1_pos.shape[0] == 1:
                """以下只对只有1个L4且只有1个S1的情况有效。采用减票数的方法，去掉那些不对的。"""
                y1 = L5_boxes[:, 0]  # 这些L5的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
                y1_L4 = L4_boxes[:, 0][0]
                y1_S1 = S1_boxes[:, 0][0]
                x1 = L5_boxes[:, 1]  # 这些L5的外接矩形的左上角点x坐标。
                x1_L4 = L4_boxes[:, 1][0]
                x1_S1 = S1_boxes[:, 1][0]
                incorrect1 = np.where(np.logical_or((y1 < y1_L4), (y1 > y1_S1)))
                # 上句，L5的话，如果发现y坐标比L4的还小，或者比S1的还大，那肯定是错的。
                L5_scores_selection[incorrect1] = L5_scores_selection[incorrect1] - 2  # 所以，减2票
                too_right = np.logical_and((x1-x1_L4 > 30), (x1-x1_S1 > 30))  # 集合逻辑运算，需要用np.logical_and。
                too_left = np.logical_and((x1-x1_L4 < -30), (x1-x1_S1 < -30))
                incorrect2 = np.where(np.logical_or(too_right, too_left))
                # 上句，那个x1[j]-x1_L4和x1[j]-x1_S1，一般情况下应该是一正一负的， 所以同时为正或者同时为负，且正或者负了那么多，就很可能是不对的。
                L5_scores_selection[incorrect2] = L5_scores_selection[incorrect2] - 1  # 所以，减1票
            L5_area_normed = (L5_boxes[:, 2] - L5_boxes[:, 0]) * (L5_boxes[:, 3] - L5_boxes[:, 1]) / 512 / 512  # 512日后改成h/w
            too_small = (L5_area_normed < 0.001)  # 一般而言，椎骨的面积是在总面积的0.01左右，所以，小于10倍或者大于20倍就认为他不对。
            too_large = (L5_area_normed > 0.2)
            somewhat_small = (L5_area_normed < 0.002)  # 后来加上，小于1/5或者大于5倍，就是小于0.002或者大于0.05，也是有点不对的，不减2票，但是可以减1票。
            somewhat_large = (L5_area_normed > 0.05)
            incorrect_by_area = np.where(np.logical_or(too_small, too_large))
            maybe_incorrect_by_area = np.where(np.logical_or(somewhat_small, somewhat_large))
            L5_scores_selection[maybe_incorrect_by_area] = L5_scores_selection[maybe_incorrect_by_area] - 1  # 减1票（可能不对的）
            L5_scores_selection[incorrect_by_area] = L5_scores_selection[incorrect_by_area] - 1  # 再减1票，一共减2票（因为上面靠距离加的票，最多是1.5票）
            L5_scores_selection_sorted_pos = np.argsort(L5_scores_selection)  # 投票结果从小到大排列的顺序
            L5_selected = L5_scores_selection_sorted_pos[-1]  # 选最后一个，不确定行不行呢。。。
            L5_rois_selected = L5_boxes[L5_selected]
            L5_scores_selected = L5_scores[L5_selected]
            L5_masks_selected = L5_masks[:, :, L5_selected]
            L5_masks_selected = np.expand_dims(L5_masks_selected, axis=2)
        else:
            L5_rois_selected = L5_boxes[0]
            L5_scores_selected = L5_scores[0]
            L5_masks_selected = L5_masks
        """以上通过投票，选出来了最高分值的L4、L5、S1，现在把它们拼起来组成最后的输出。"""
        final_rois_refined = np.stack((L4_rois_selected, L5_rois_selected, S1_rois_selected), axis=0)
        final_class_ids_refined = np.array([1, 2, 3])  # 因为是这三种一种一个，所以就直接就是1 2 3了。
        final_scores_refined = np.stack((L4_scores_selected, L5_scores_selected, S1_scores_selected), axis=0)
        final_masks_refined = np.concatenate((L4_masks_selected, L5_masks_selected, S1_masks_selected), axis=2)  # 这个的axis应该是2...
    else:
        print('有的椎骨没检测出来。怎么输入怎么输出了。')
        final_rois_refined = final_rois
        final_class_ids_refined = final_class_ids
        final_scores_refined = final_scores
        final_masks_refined = final_masks
    return final_rois_refined, final_class_ids_refined, final_scores_refined, final_masks_refined

def main(_):
    matfn = 'E:/赵屾的文件/55-脊柱滑脱/Spine-Gan-plus-MRCNN/detectiond_with_multi_L4S1_new.mat'  # 里面2张图，一张是多了L4，一张是多了S1
    # matfn = 'E:/赵屾的文件/55-脊柱滑脱/Spine-Gan-plus-MRCNN/detectiond_with_multi_L4S1_new1.mat'  # 里面1张图，多了L4，但是是跑到别的地方去了。
    data = sio.loadmat(matfn)
    # detections = data['detections']  # 2张图的
    which_image = 1  # mat文件里就两张图，拿来试试玩的
    assert which_image in [0, 1]
    if which_image == 0:
        final_rois = data['final_rois']  # 1张图的！不要加[0]
        final_class_ids = data['final_class_ids'][0]  # 1张图的！要加[0]
        final_scores = data['final_scores'][0]  # 1张图的！要加[0]
        final_masks = data['final_masks']  # 1张图的！不要加[0]
    else:
        final_rois = data['final_rois1']  # 1张图的！不要加[0]
        final_class_ids = data['final_class_ids1'][0]  # 1张图的！要加[0]
        final_scores = data['final_scores1'][0]  # 1张图的！要加[0]
        final_masks = data['final_masks1']  # 1张图的！不要加[0]
    # PS：以上提出，对应的真实病人序号是： [256 154]
    final_rois_confirmed, final_class_ids_confirmed, final_scores_confirmed, final_masks_confirmed = \
        delete_incorrect_L4_and_S1(final_rois, final_class_ids, final_scores, final_masks)
    print('final_rois_confirmed', final_rois_confirmed)
    print('final_class_ids_confirmed', final_class_ids_confirmed)
    print('final_scores_confirmed', final_scores_confirmed)
    # print(final_masks_confirmed)

if __name__ == '__main__':
    tf.app.run()