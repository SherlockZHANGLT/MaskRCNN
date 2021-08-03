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
    """处理一张图中，可能的三种情况：
    现在能处理三种情况：
    1、L5只有1个，L4多了；
    2、L5只有1个，S1多了；
        （当然后来发现，如果L5只有1个，而L4和S1都多了的情况，也可以处理）
    3、L4和S1都只有1个，L5多了
    所以说，这个程序其实只能处理MRCNN结果已经比较好了的情况。如果弄出来的东西都乱七八糟的，就没用了。
    【后来(180925)感觉，分这三种情况处理，是可以的，但是，好像没必要这么麻烦吧。因为现在通过加GAN、规定target_class_ids数量、正负例个数等措施，
    已经可以把测试阶段的提出弄得还好了。所以，为何不在这三种情况下，直接根据位置、分数去做投票，然后直接选择得票最高的呢？！】
    """
    L4_pos = np.array(np.where(final_class_ids == 1))[0]  # array([2], dtype=int64)  这个地方各种加[0]什么的特别烦
    L5_pos = np.array(np.where(final_class_ids == 2))[0]
    S1_pos = np.array(np.where(final_class_ids == 3))[0]  # array([1, 3], dtype=int64)
    L4_boxes = final_rois[L4_pos, :]  # 那些被认为是L4的提出
    L5_boxes = final_rois[L5_pos, :]  # 那些被认为是L5的提出
    S1_boxes = final_rois[S1_pos, :]  # 那些被认为是S1的提出
    """注意，本程序只有在提出框已经比较好的时候才适用！！！"""
    if L4_pos.shape[0] > 1 and L5_pos.shape[0] == 1:
        print('检测到了多于一个L4，不过好在L5只有一个。')
        while (L4_pos.shape[0] > 1):
            y1 = L4_boxes[:, 0]  # 这些L4-L5的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
            x1_L4 = L4_boxes[:, 1]
            x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
            """以下是第一个环节，观察是否有肯定对的。"""
            down_ix = np.argmax(y1, axis=0)  # 只要最靠下的那个
            down_L4 = L4_boxes[down_ix, :]  # 最靠上方的那个，似乎一般它就是真的那个L4。
            up_L4_dist = np.mean(abs(L5_boxes - down_L4), axis=1)  # 和L5的距离（用角点坐标差值表示）。后改成角点坐标差值绝对值。
            up_L4_dist = np.squeeze(up_L4_dist)
            mean_dist = np.mean(abs(L5_boxes - L4_boxes), axis=1)  # 那些L4的外接矩形，和L5的外接矩形的四个坐标点的平均距离
            min_mean_dist = np.min(mean_dist)
            overlap_with_L5 = compute_overlaps(np.expand_dims(down_L4, axis=0), L5_boxes)
            overlap_with_L5 = np.squeeze(overlap_with_L5)
            if overlap_with_L5 < 0.25 and up_L4_dist == min_mean_dist \
                    and final_scores[L4_pos[down_ix]] > 0.9 and abs(x1_L4[down_ix]-x1_L5) < 50:
                """
                以上4个条件：和L5重合不能太多（否则把L5当成L4也悲剧）、离L5最近、预测信心分数大于0.9、和L5的x方向坐标不能太远（小于50）
                    最后一个条件不确定，万一真的滑脱的，就是差了这么远，该怎么办？
                    第二个也有点问题，如果就是有一个不对的，但是距离就是比那个对的要近，怎么办？
                原来是，先找到最靠下的那个L4，看看对不对，如果对，直接确定下来。其实都觉得这段都没什么必要了，
                因为后面的那个循环可以直接一个循环过去，删掉不想要的那些。。
                """
                confirmed_L4 = down_L4
                L4_pos_confirmed = L4_pos[down_ix]  # 要保留的那个，在输入的final_rois中是第几个，是一个int64的数
                L4_pos_delete = set(L4_pos) - {L4_pos_confirmed}  # 集合相减（那个{}是把int64的数变成集合），得到final_rois中要删除的行
                L4_pos_delete = [i for i in L4_pos_delete]  # 集合变成list
                for i in range(len(L4_pos_delete)):
                    final_rois = np.delete(final_rois, L4_pos_delete[0], axis=0)  # 删除final_rois里对应的行
                    final_class_ids = np.delete(final_class_ids, L4_pos_delete[0], axis=0)  # 删除final_class_ids里对应的行
                    final_scores = np.delete(final_scores, L4_pos_delete[0], axis=0)
                    final_masks = np.delete(final_masks, L4_pos_delete[0], axis=2)  # 删除对应的掩膜，注意现在axis是2而其他的是0。
                # 需要break么？好像可以，因为是把别的都删了。。。不过不break似乎也行，因为都删了的话，反正再循环过来，while条件就不满足了。。
                print('找到满足四个条件（和L5重合不太多、离L5最近、信心分数大于0.9、和L5的x坐标差得不太远）的L4，确定下来了。')
                break
            else:
                """以下是第二个环节，观察是否有肯定不对的"""
                to_delete = []
                for i in range(L4_boxes.shape[0]):
                    L4 = L4_boxes[i, :]
                    overlap_with_L5 = compute_overlaps(np.expand_dims(L4, axis=0), L5_boxes)
                    overlap_with_L5 = np.squeeze(overlap_with_L5)
                    if overlap_with_L5 > 0.5 or final_scores[L4_pos[i]] < 0.7 or abs(x1_L4[i]-x1_L5) > 50:
                        """以上几种情况，应该可以确定这个提出应该是不对的。即，要么和L5重合太多，要么分数太低，要么横向偏得太远。"""
                        print("发现和L5重合太多、或分数太低、或横向偏得太远，删除这个提出。")
                        to_delete.append(i)
                for id in to_delete:
                    L4_pos_delete = L4_pos[id]
                    final_rois = np.delete(final_rois, L4_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, L4_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, L4_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, L4_pos_delete, axis=2)
                    L4_boxes = np.delete(L4_boxes, id, axis=0)
                    L4_pos = np.delete(L4_pos, id, axis=0)
            if L4_pos.shape[0] > 1:
                """如果到这时候了，还是不止1个L4，就采用投票法选出来1个。"""
                L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 先更新一下
                L5_pos = np.array(np.where(final_class_ids == 2))[0]
                L4_boxes = final_rois[L4_pos, :]  # 那些被认为是L4的提出
                L5_boxes = final_rois[L5_pos, :]  # 那些被认为是L5的提出
                L4_score = final_scores[L4_pos]  # L4的信心分数
                scores = np.zeros([L4_pos.shape[0]])  # 找提出的分数
                # x1_L4 = L4_boxes[:, 1]
                # x1_L5 = L5_boxes[:, 1][0]
                # x2_L4 = L4_boxes[:, 3]
                # x2_L5 = L5_boxes[:, 3][0]
                # y1_L4 = L4_boxes[:, 0]
                # y1_L5 = L5_boxes[:, 0][0]
                # y2_L4 = L4_boxes[:, 2]
                # y2_L5 = L5_boxes[:, 2][0]
                # x1_diff = abs(x1_L5-x1_L4)
                # x2_diff = abs(x2_L5-x2_L4)
                # y1_diff = abs(y1_L5 - y1_L4)
                # y2_diff = abs(y2_L5 - y2_L4)
                diff = abs(L4_boxes - L5_boxes)
                mean_diff = np.mean(diff, axis=1)
                min_diff_pos = np.argmin(mean_diff, axis=0)
                scores[min_diff_pos] = scores[min_diff_pos] + 1
                max_score = np.max(L4_score, axis=0)
                max_score_pos = np.argmax(L4_score, axis=0)
                L4_score_others = np.delete(L4_score, max_score_pos, axis=0)
                max_L4_score_others = np.max(L4_score_others)
                if max_score >= max_L4_score_others+0.01:  # 暂定0.01这个阈值
                    scores[max_score_pos] = scores[max_score_pos] + 1
                scores_max = np.max(scores, axis=0)  # 找提出的最高得分
                selected = np.array(np.where(scores==scores_max))[0]
                assert selected.shape[0]>=1
                if selected.shape[0]==1:
                    L4_pos_selected = L4_pos[selected]
                else:  # 如果不幸地，发现两个得分一样高的
                    L4_pos_selected = L4_pos[min_diff_pos]  # 就选离L5最小的了。
                L4_pos_delete = set(L4_pos) - set(L4_pos_selected)  # 集合相减（那个{}是把int64的数变成集合），得到final_rois中要删除的行
                L4_pos_delete = [i for i in L4_pos_delete]  # 集合变成list
                L4_pos = L4_pos_selected  # 更新一下这俩，让他能跳出去while循环
                L4_boxes = L4_boxes[selected]
                for i in range(len(L4_pos_delete)):
                    final_rois = np.delete(final_rois, L4_pos_delete[0], axis=0)  # 删除final_rois里对应的行
                    final_class_ids = np.delete(final_class_ids, L4_pos_delete[0], axis=0)  # 删除final_class_ids里对应的行
                    final_scores = np.delete(final_scores, L4_pos_delete[0], axis=0)
                    final_masks = np.delete(final_masks, L4_pos_delete[0], axis=2)  # 删除对应的掩膜，注意现在axis是2而其他的是0。

    L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 得更新一下那些位置（至少L5和S1的得更新），因为前面可能删除了final_class_ids等矩阵中的一些行。
    L5_pos = np.array(np.where(final_class_ids == 2))[0]
    S1_pos = np.array(np.where(final_class_ids == 3))[0]  #
    L4_boxes = final_rois[L4_pos, :]  #
    L5_boxes = final_rois[L5_pos, :]  #
    S1_boxes = final_rois[S1_pos, :]  #

    if S1_pos.shape[0] > 1 and L5_pos.shape[0] == 1:  # 如果S1多了，而L5正好是1个。一般情况下，多出来的S1会在正常的S1右下方（比如，把S2也误认为是S1了）
        print('检测到了多于一个S1，不过好在L5只有一个。')
        """以下是第一个环节，观察是否有肯定对的。"""
        while (S1_pos.shape[0] > 1):
            y1 = S1_boxes[:, 0]  # 这些S1的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
            x1_S1 = S1_boxes[:, 1]
            x2_S1 = S1_boxes[:, 3]
            x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
            x2_L5 = L5_boxes[:, 3][0]  # 加一个[0]把1维的array变成数
            up_ix = np.argmin(y1, axis=0)  # 只要最靠上的那个
            up_S1 = S1_boxes[up_ix, :]  # 最靠上方的那个，似乎一般它就是真的那个S1。
            up_S1_dist = np.mean(abs(up_S1 - L5_boxes), axis=1)  # 和L5的距离（用角点坐标差值表示）。后改成角点坐标差值绝对值。
            up_S1_dist = np.squeeze(up_S1_dist)
            mean_dist = np.mean(abs(S1_boxes - L5_boxes), axis=1)  # 那些S1的外接矩形，和L5的外接矩形的四个坐标点的平均距离
            min_mean_dist = np.min(mean_dist)
            overlap_with_L5 = compute_overlaps(np.expand_dims(up_S1, axis=0), L5_boxes)
            overlap_with_L5 = np.squeeze(overlap_with_L5)
            if overlap_with_L5 < 0.25 and up_S1_dist == min_mean_dist and \
                    final_scores[S1_pos[up_ix]] > 0.9 and abs(x1_S1[up_ix]-x1_L5) < 50:
                """原来是，先找到最靠上的那个S1，看看对不对，如果对，直接确定下来。其实都觉得这段都没什么必要了，
                因为后面的那个循环可以直接一个循环过去，删掉不想要的那些。。"""
                confirmed_S1 = up_S1
                S1_pos_confirmed = S1_pos[up_ix]  # 要保留的那个，在输入的final_rois中是第几个，是一个int64的数
                S1_pos_delete = set(S1_pos) - {S1_pos_confirmed}  # 集合相减（那个{}是把int64的数变成集合），得到final_rois中要删除的行
                S1_pos_delete = [i for i in S1_pos_delete]  # 集合变成list
                for i in range(len(S1_pos_delete)):
                    final_rois = np.delete(final_rois, S1_pos_delete[0], axis=0)  # 删除final_rois里对应的行
                    final_class_ids = np.delete(final_class_ids, S1_pos_delete[0], axis=0)  # 删除final_class_ids里对应的行
                    final_scores = np.delete(final_scores, S1_pos_delete[0], axis=0)
                    final_masks = np.delete(final_masks, S1_pos_delete[0], axis=2)  # 删除对应的掩膜，注意现在axis是2而其他的是0。
                print('找到满足四个条件（和L5重合不太多、离L5最近、信心分数大于0.9、和L5的x坐标差得不太远）的S1，确定下来了。')
                break
            else:
                """以下是第二个环节，观察是否有肯定不对的"""
                to_delete = []
                for i in range(S1_boxes.shape[0]):
                    S1 = S1_boxes[i, :]
                    overlap_with_L5 = compute_overlaps(np.expand_dims(S1, axis=0), L5_boxes)
                    overlap_with_L5 = np.squeeze(overlap_with_L5)
                    if overlap_with_L5 > 0.5 or final_scores[S1_pos[i]] < 0.7 or abs(x1_S1[i] - x1_L5) > 50 or abs(x2_S1[i] - x2_L5) > 50:
                        """以上几种情况，应该可以确定这个提出应该是不对的。即，要么和L5重合太多，要么分数太低，要么横向偏得太远。"""
                        print("发现和L5重合太多、或分数太低、或横向偏得太远，删除这个提出。")
                        to_delete.append(i)
                for id in to_delete:
                    S1_pos_delete = S1_pos[id]
                    final_rois = np.delete(final_rois, S1_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, S1_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, S1_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, S1_pos_delete, axis=2)
                    S1_boxes = np.delete(S1_boxes, id, axis=0)
                    S1_pos = np.delete(S1_pos, id, axis=0)
            if S1_pos.shape[0] > 1:
                """如果到这时候了，还是不止1个S1，就采用投票法选出来1个。"""
                S1_pos = np.array(np.where(final_class_ids == 1))[0]  # 先更新一下
                L5_pos = np.array(np.where(final_class_ids == 2))[0]
                S1_boxes = final_rois[S1_pos, :]  # 那些被认为是S1的提出
                L5_boxes = final_rois[L5_pos, :]  # 那些被认为是L5的提出
                S1_score = final_scores[S1_pos]  # S1的信心分数
                scores = np.zeros([S1_pos.shape[0]])  # 找提出的分数
                # x1_S1 = S1_boxes[:, 1]
                # x1_L5 = L5_boxes[:, 1][0]
                # x2_S1 = S1_boxes[:, 3]
                # x2_L5 = L5_boxes[:, 3][0]
                # y1_S1 = S1_boxes[:, 0]
                # y1_L5 = L5_boxes[:, 0][0]
                # y2_S1 = S1_boxes[:, 2]
                # y2_L5 = L5_boxes[:, 2][0]
                # x1_diff = abs(x1_L5-x1_S1)
                # x2_diff = abs(x2_L5-x2_S1)
                # y1_diff = abs(y1_L5 - y1_S1)
                # y2_diff = abs(y2_L5 - y2_S1)
                diff = abs(S1_boxes - L5_boxes)
                mean_diff = np.mean(diff, axis=1)
                min_diff_pos = np.argmin(mean_diff, axis=0)
                scores[min_diff_pos] = scores[min_diff_pos] + 1
                max_score = np.max(S1_score, axis=0)
                max_score_pos = np.argmax(S1_score, axis=0)
                S1_score_others = np.delete(S1_score, max_score_pos, axis=0)
                max_S1_score_others = np.max(S1_score_others)
                if max_score >= max_S1_score_others + 0.01:  # 暂定0.01这个阈值
                    scores[max_score_pos] = scores[max_score_pos] + 1
                scores_max = np.max(scores, axis=0)  # 找提出的最高得分
                selected = np.array(np.where(scores == scores_max))[0]
                assert selected.shape[0] >= 1
                if selected.shape[0] == 1:
                    S1_pos_selected = S1_pos[selected]
                else:  # 如果不幸地，发现两个得分一样高的
                    S1_pos_selected = S1_pos[min_diff_pos]  # 就选离L5最小的了。
                S1_pos_delete = set(S1_pos) - set(S1_pos_selected)  # 集合相减（那个{}是把int64的数变成集合），得到final_rois中要删除的行
                S1_pos_delete = [i for i in S1_pos_delete]  # 集合变成list
                S1_pos = S1_pos_selected  # 更新一下这俩，让他能跳出去while循环
                S1_boxes = S1_boxes[selected]
                for i in range(len(S1_pos_delete)):
                    final_rois = np.delete(final_rois, S1_pos_delete[0], axis=0)  # 删除final_rois里对应的行
                    final_class_ids = np.delete(final_class_ids, S1_pos_delete[0], axis=0)  # 删除final_class_ids里对应的行
                    final_scores = np.delete(final_scores, S1_pos_delete[0], axis=0)
                    final_masks = np.delete(final_masks, S1_pos_delete[0], axis=2)  # 删除对应的掩膜，注意现在axis是2而其他的是0。

    L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 继续更新一下那些位置（至少L5和S1的得更新），因为前面可能删除了final_class_ids等矩阵中的一些行。
    L5_pos = np.array(np.where(final_class_ids == 2))[0]
    S1_pos = np.array(np.where(final_class_ids == 3))[0]  #
    L4_boxes = final_rois[L4_pos, :]  #
    L5_boxes = final_rois[L5_pos, :]  #
    S1_boxes = final_rois[S1_pos, :]  #

    if L5_pos.shape[0] > 1 and L4_pos.shape[0] == 1 and S1_pos.shape[0] == 1:  # 如果L5多了，而L4/S1都正好是1个。
        """这种情况，一般就认为"""
        print('检测到了多于一个L5。还好L4和S1都只有1个，处理一下试试。')
        while (L5_pos.shape[0] > 1):
            """换一种方法，排除法：
            1、L5的话，如果发现y坐标比L4的还小，或者比S1的还大，那肯定是错的。
            2、如果x坐标和L4和S1的平均距离差太多，那肯定是错的（就算滑脱，也不至于这样的）。
            """
            y1 = L5_boxes[:, 0]  # 这些L5的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
            y1_L4 = L4_boxes[:, 0][0]
            y1_S1 = S1_boxes[:, 0][0]
            x1 = L5_boxes[:, 1]  # 这些L5的外接矩形的左上角点x坐标。
            x1_L4 = L4_boxes[:, 1][0]
            x1_S1 = S1_boxes[:, 1][0]
            for j in range(len(y1)):
                if y1[j] < y1_L4 or y1[j] > y1_S1 or \
                    (x1[j]-x1_L4 > 30 and x1[j]-x1_S1 > 30) or (x1[j]-x1_L4 < -30 and x1[j]-x1_S1 < -30):
                    """那个x1[j]-x1_L4和x1[j]-x1_S1，一般情况下应该是一正一负的，
                    所以同时为正或者同时为负，且正或者负了那么多，就很可能是不对的"""
                    print('发现这个L5y坐标跑到L4上边或者S1下边，或者横坐标在L4和S1左/右边太多')
                    L5_pos_delete = L5_pos[j]
                    final_rois = np.delete(final_rois, L5_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, L5_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, L5_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, L5_pos_delete, axis=2)
                    L5_boxes = np.delete(L5_boxes, j, axis=0)
                    L5_pos = np.delete(L5_pos, j, axis=0)
    if L5_pos.shape[0] > 1 and (L4_pos.shape[0] == 1 and S1_pos.shape[0] > 1):  # 如果L5和S1都多了，而L4都正好是1个。
        print('检测到了多于一个L5和多于1个的S1。L4只有1个，就认为L4是对的。')
        while (L5_pos.shape[0] > 1):
            """换一种方法，排除法：
            1、L5的话，如果发现y坐标比L4的还小，那肯定是错的。
            2、如果x坐标和L4的平均距离差太多，那肯定是错的（就算滑脱，也不至于这样的）。
            比起L4和S1都只有1个的，这儿就是只用L4的信息了。
            """
            y1 = L5_boxes[:, 0]  # 这些L5的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
            y1_L4 = L4_boxes[:, 0][0]
            x1 = L5_boxes[:, 1]  # 这些L5的外接矩形的左上角点x坐标。
            x2 = L5_boxes[:, 3]  # 这些L5的外接矩形的右下角点x坐标。
            x1_L4 = L4_boxes[:, 1][0]
            x2_L4 = L4_boxes[:, 3][0]
            for j in range(len(y1)):
                if y1[j] < y1_L4 or  \
                    (x1[j]-x1_L4 > 30 and x2[j]-x2_L4 > 30) or (x1[j]-x1_L4 < -30 and x2[j]-x2_L4 < -30):
                    print('发现这个L5y坐标跑到L4上边，或者横坐标在L4左/右边太多')
                    L5_pos_delete = L5_pos[j]
                    final_rois = np.delete(final_rois, L5_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, L5_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, L5_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, L5_pos_delete, axis=2)
                    L5_boxes = np.delete(L5_boxes, j, axis=0)
                    L5_pos = np.delete(L5_pos, j, axis=0)
        L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 继续更新一下那些位置（至少L5和S1的得更新），因为前面可能删除了final_class_ids等矩阵中的一些行。
        L5_pos = np.array(np.where(final_class_ids == 2))[0]
        S1_pos = np.array(np.where(final_class_ids == 3))[0]  #
        L4_boxes = final_rois[L4_pos, :]  #
        L5_boxes = final_rois[L5_pos, :]  #
        S1_boxes = final_rois[S1_pos, :]  #
        while (S1_pos.shape[0] > 1):  # 处理S1的事情（当然，这儿是假设L5已经搞定了的。。）
            x1_S1 = S1_boxes[:, 1]
            x2_S1 = S1_boxes[:, 3]
            x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
            x2_L5 = L5_boxes[:, 3][0]  # 加一个[0]把1维的array变成数
            for i in range(S1_boxes.shape[0]):
                S1 = S1_boxes[i, :]
                overlap_with_L5 = compute_overlaps(np.expand_dims(S1, axis=0), L5_boxes)
                overlap_with_L5 = np.squeeze(overlap_with_L5)
                if overlap_with_L5 > 0.5 or final_scores[S1_pos[i]] < 0.7 or abs(x1_S1[i] - x1_L5) > 50 or abs(x2_S1[i] - x2_L5) > 50:
                    """以上几种情况，应该可以确定这个提出应该是不对的。即，要么和L5重合太多，要么分数太低，要么横向偏得太远。"""
                    print("发现和L5重合太多、或分数太低、或横向偏得太远，删除这个提出。")
                    S1_pos_delete = S1_pos[i]
                    final_rois = np.delete(final_rois, S1_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, S1_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, S1_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, S1_pos_delete, axis=2)
                    S1_boxes = np.delete(S1_boxes, i, axis=0)
                    S1_pos = np.delete(S1_pos, i, axis=0)
    if L5_pos.shape[0] > 1 and (L4_pos.shape[0] > 1 and S1_pos.shape[0] == 1):  # 如果L5多了，而S1都正好是1个。
        print('检测到了多于一个L5和多于1个的S1。L4只有1个，就认为L4是对的。')
        while (L5_pos.shape[0] > 1):
            """换一种方法，排除法：
            1、L5的话，如果发现y坐标比L4的还小，那肯定是错的。
            2、如果x坐标和L4的平均距离差太多，那肯定是错的（就算滑脱，也不至于这样的）。
            比起L4和S1都只有1个的，这儿就是只用L4的信息了。
            """
            y1 = L5_boxes[:, 0]  # 这些L5的外接矩形的左上角点y坐标。注意那个外接矩形的坐标是y1, x1, y2, x2！！
            y1_S1 = S1_boxes[:, 0][0]
            x1 = L5_boxes[:, 1]  # 这些L5的外接矩形的左上角点x坐标。
            x2 = L5_boxes[:, 3]  # 这些L5的外接矩形的右下角点x坐标。
            x1_S1 = S1_boxes[:, 1][0]
            x2_S1 = S1_boxes[:, 3][0]
            for j in range(len(y1)):
                if y1[j] > y1_S1 or  \
                    (x1[j]-x1_S1 > 30 and x2[j]-x2_S1 > 30) or (x1[j]-x1_S1 < -30 and x2[j]-x2_S1 < -30):
                    print('发现这个L5y坐标跑到S1下边，或者横坐标在S1左/右边太多')
                    L5_pos_delete = L5_pos[j]
                    final_rois = np.delete(final_rois, L5_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, L5_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, L5_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, L5_pos_delete, axis=2)
                    L5_boxes = np.delete(L5_boxes, j, axis=0)
                    L5_pos = np.delete(L5_pos, j, axis=0)
        L4_pos = np.array(np.where(final_class_ids == 1))[0]  # 继续更新一下那些位置（至少L5和S1的得更新），因为前面可能删除了final_class_ids等矩阵中的一些行。
        L5_pos = np.array(np.where(final_class_ids == 2))[0]
        S1_pos = np.array(np.where(final_class_ids == 3))[0]  #
        L4_boxes = final_rois[L4_pos, :]  #
        L5_boxes = final_rois[L5_pos, :]  #
        S1_boxes = final_rois[S1_pos, :]  #
        while (S1_pos.shape[0] > 1):  # 处理L4的事情（当然，这儿是假设L5已经搞定了的。。）
            x1_L4 = L4_boxes[:, 1]
            x2_L4 = L4_boxes[:, 3]
            x1_L5 = L5_boxes[:, 1][0]  # 加一个[0]把1维的array变成数
            x2_L5 = L5_boxes[:, 3][0]  # 加一个[0]把1维的array变成数
            for i in range(L4_boxes.shape[0]):
                L4 = L4_boxes[i, :]
                overlap_with_L5 = compute_overlaps(np.expand_dims(L4, axis=0), L5_boxes)
                overlap_with_L5 = np.squeeze(overlap_with_L5)
                if overlap_with_L5 > 0.5 or final_scores[L4_pos[i]] < 0.7 or abs(x1_L4[i] - x1_L5) > 50 or abs(x2_L4[i] - x2_L5) > 50:
                    """以上几种情况，应该可以确定这个提出应该是不对的。即，要么和L5重合太多，要么分数太低，要么横向偏得太远。"""
                    print("发现和L5重合太多、或分数太低、或横向偏得太远，删除这个提出。")
                    L4_pos_delete = L4_pos[i]
                    final_rois = np.delete(final_rois, L4_pos_delete, axis=0)
                    final_class_ids = np.delete(final_class_ids, L4_pos_delete, axis=0)
                    final_scores = np.delete(final_scores, L4_pos_delete, axis=0)
                    final_masks = np.delete(final_masks, L4_pos_delete, axis=2)
                    L4_boxes = np.delete(L4_boxes, i, axis=0)
                    L4_pos = np.delete(L4_pos, i, axis=0)
    return final_rois, final_class_ids, final_scores, final_masks

def main(_):
    # matfn = 'E:/赵屾的文件/55-脊柱滑脱/Spine-Gan-plus-MRCNN/detectiond_with_multi_L4S1_new.mat'  # 里面2张图，一张是多了L4，一张是多了S1
    matfn = 'E:/赵屾的文件/55-脊柱滑脱/Spine-Gan-plus-MRCNN/detectiond_with_multi_L4S1_new1.mat'  # 里面1张图，多了L4，但是是跑到别的地方去了。
    data = sio.loadmat(matfn)
    # detections = data['detections']  # 2张图的
    which_image = 0  # mat文件里就两张图，拿来试试玩的
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