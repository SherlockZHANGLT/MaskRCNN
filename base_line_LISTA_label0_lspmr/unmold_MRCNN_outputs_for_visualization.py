import numpy as np
import MRCNN_utils as utils
def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.
    以下输入的所有东西都是np或者tuple，反正没tf。
    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]，
        跑的时候是(100, 28, 28, 7)，这是说这张图里有100个提出，每个提出都是28*28的，然后有7种类别。
    original_image_shape: [H, W, C] Original image shape before resizing
        跑的时候是(512, 512, 3)
    image_shape: [H, W, C] Shape of the image after resizing and padding
        跑的时候也是(512, 512, 3)
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real image is excluding the padding.
        跑的时候是array([  0,   0, 512, 512])
    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        以像素点为单位的外接矩形。验证了一下，本函数被调用的时候，输入的_detections[i]和输出的final_rois之间，其实大多数也就是差了个512倍。
            也就是说，那个_detections的前4个数就是预测的外接矩形的归一化了的坐标，如果乘以512，基本上就是预测的外接矩形了。
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]  # 0~1之间的数，就是归一化坐标的外接矩形了。
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    # masks的shape变成了(24, 28, 28)，因为一方面是去掉了补零，
    # 一方面是在每个提出的7个掩膜里，选了分类对应的那个掩膜，比如说如果分类的时候判断这个掩膜属于第1类，那就只要第1个掩膜，其他的都不要。
    # 现在每个提出都有个28*28的掩膜，其中每个点的数值都是0~1之间的数，应该就代表这个点属于这一类（如，刚才所说的第1类）的概率。

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])  # window是[0 0 1 1]
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height 是1
    ww = wx2 - wx1  # window width 也是1
    scale = np.array([wh, ww, wh, ww])  # scale是[1 1 1 1]
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])  # 现在boxes是实体坐标的、外接矩形角点。
    # 比如说，如果boxes[0]是[301, 240, 344, 255]，那就是说第0个外接矩形最上边是301，最下边是244，最左是240，最右是255。
    if np.max(boxes)>512:
        print ('发现超出边界的提出框')
        boxes[boxes>512] = 512  # 超出图像边界的部分，裁剪到512（似乎是没有<0的，所以就不管他了）；第58或154张图出了这个问题。。

    # Filter out detections with zero area. Happens in early training when network weights are still random
    # 过滤掉面积为0的探测。
    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]  # 现在是没有。
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        try:
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            # 输入masks[i]的shape是(28, 28)，是在处理1个提出的掩膜；boxes[i]应该是这个提出的角点；original_image_shape是(512, 512, 3)，原图大小；
            # 输出的full_mask是(512, 512)的，在这个函数里面进行了缩放，还有就是把概率按照阈值变成0或1，并且放到原图（大小为512,512）上，得到最终的掩膜。
            full_masks.append(full_mask)
        except:
            print('似乎有点问题，手动试着执行一下unmold_mask。如果这种情况，后续full_masks可能会是0。')
    full_masks = np.stack(full_masks, axis=-1) \
        if full_masks else np.empty(masks.shape[1:3] + (0,))  # stack完了之后，得到shape是(512, 512, 24)。

    return boxes, class_ids, scores, full_masks


def judge_empty_for_tf(input):
    """
    判断输入（一个shape=(?,4)的张量，当然现在是要弄成np.array的形式）是否为空（即那个?是否为0），
    如果是空的，就返回shape=(4,)的0（即4个0）；
    如果不是空的，就返回输入张量的外包络（即第0和2维取最小值，第1和3维取最大值，也返回shape=(4,)的张量）。
    感觉这个东西会比较常用，尽管输出的东西未必一样。。。
    """
    if input.size == 0:
        output = np.zeros([4,])
        return output
    else:
        y1min = np.min(input[:, 0])
        y2max = np.max(input[:, 2])
        x1min = np.min(input[:, 1])
        x2max = np.max(input[:, 3])
        output = np.array([y1min, x1min, y2max, x2max])  # 外包络，相当于input是若干个矩形，output是所有这些矩形的“总的”矩形。
        return output


def unmold_detections_for_GAN(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """这个函数和上面的那个基本上一样，也是用np写的，只不过在MaskRCNN_1_model.py里，用这个函数作为tf.py_func用的。
    1、除了返回上面函数返回的那些之外，还返回了mask_for_GAN，一个(512, 512, 7)的np.array，回到MaskRCNN_1_model.py里就是个张量。
    2、增加了边界裁切，如果外接矩形大于512，就裁剪到512去（上面那个函数后来也这么做了）。
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    # print('开始执行。。')
    zero_ix = np.where(detections[:, 4] == 0)[0]  # zero_ix是detections中第一个为0的行的序号。detections的shape是(100, 6)，100即补零后的提出数。
    #     所以，如果有20个非零提出，那么zero_ix就是20~99，后面zero_ix.shape[0]就是80。
    # print ('detections中为0的行的序号：' , zero_ix.shape, 'zero_ix的值是：', zero_ix)
    # print ('detections.shape是', detections.shape)
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    # 上句，如果zero_ix.shape[0] > 0（那1100个提出有非零的），那么N就是非零提出数；如果没有（刚开始训练的时候可能这样），那么就认为是100个。
    class_numbers = mrcnn_mask.shape[3]  # 一共多少种类别
    # print("一开始N值是%d" % N)
    if N == 0:
        print('N=0！是因为一开始训练的时候没有找到，还是别的情况？暂且返回5个全零的提出，然后mask_for_GAN就认为背景类的概率都是0.999，其他类都是0.001。')
        # 果然发现，一开始训练的时候，有时候N值会是0，然后这儿就会报错说
        #   “InvalidArgumentError: ValueError: zero-size array to reduction operation maximum which has no identity”。
        boxes = detections[:5, :4].astype(np.int32)  # 0~1之间的数，就是归一化坐标的外接矩形了。第一个:5就是说这种时候给他返回5个0去。。
        class_ids = detections[:5, 4].astype(np.int32)
        scores = detections[:5, 5]
        full_masks = np.zeros([original_image_shape[0], original_image_shape[1], 5], dtype=np.uint8)
        mask_for_GAN = np.ones((512, 512, class_numbers), dtype=float) * 0.001  # 都初始化为0.001，就是很接近0的
        mask_for_GAN[:, :, 0] = np.ones((512, 512), dtype=float) * 0.999  # 背景类，都改成0.999，就是很接近1的
        box_L4_outer = np.zeros([4,], dtype=np.int32)  # 似乎得加这个，否则会说什么5-th value returned by pyfunc_0 is double, but expects int32（可能他是从第0个开始的，所以这个是第5个返回值）
        box_L5_outer = np.zeros([4,], dtype=np.int32)
        box_S1_outer = np.zeros([4,], dtype=np.int32)
    else:
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]  # 0~1之间的数，就是归一化坐标的外接矩形了。
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        # masks的shape变成了(24, 28, 28)，因为一方面是去掉了补零，
        # 一方面是在每个提出的7个掩膜里，选了分类对应的那个掩膜，比如说如果分类的时候判断这个掩膜属于第1类，那就只要第1个掩膜，其他的都不要。
        # 现在每个提出都有个28*28的掩膜，其中每个点的数值都是0~1之间的数，应该就代表这个点属于这一类（如，刚才所说的第1类）的概率。

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])  # window是[0 0 1 1]
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height 是1
        ww = wx2 - wx1  # window width 也是1
        scale = np.array([wh, ww, wh, ww])  # scale是[1 1 1 1]
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])  # 现在boxes是实体坐标的、外接矩形角点。比如说，如果boxes[0]是[301, 240, 344, 255]，那就是说第0个外接矩形最上边是301，最下边是244，最左是240，最右是255。

        if np.max(boxes)>512:  # 原来写的是np.max(boxes>512)，感觉别扭。可是原来做的为啥好像又没问题似的？
            print ('发现超出边界的提出框')
            boxes[boxes>512] = 512  # 超出图像边界的部分，裁剪到512（似乎是没有<0的，所以就不管他了）；第58或154张图出了这个问题。。

        # Filter out detections with zero area. Happens in early training when network weights are still random
        # 过滤掉面积为0的探测。
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]  # 现在是没有。
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        # print ('弄出来了full_masks，其实就是看tf.py_func能否执行到此处的。。现在N值是%d。' % N)
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            # 输入masks[i]的shape是(28, 28)，是在处理1个提出的掩膜；boxes[i]应该是这个提出的角点；original_image_shape是(512, 512, 3)，原图大小；
            # 输出的full_mask是(512, 512)的，在这个函数里面进行了缩放，还有就是把概率按照阈值变成0或1，并且放到原图（大小为512,512）上，得到最终的掩膜。
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(masks.shape[1:3] + (0,))  # stack完了之后，得到shape是(512, 512, 20)。那个if-else是什么鬼？？

        mask_for_GAN = np.ones((512, 512, class_numbers), dtype=float) * 0.01  # 都初始化为0.01，就是很接近0的
        mask_for_GAN[:,:,0] = np.ones((512, 512), dtype=float) * 0.99  # 背景类，都改成0.99，就是很接近1的
        # 以上，全部都是背景的一张图。
        for i in range(N):
            class_id = class_ids[i]  # 这个提出的类别，肯定不是0。
            score = scores[i]
            full_mask = full_masks[:,:,i]  # shape是(512, 512)的.
            mask_for_GAN[:,:,class_id] = full_mask * score  #
            mask_for_GAN[:, :, class_id][mask_for_GAN[:,:,class_id] < 0.01] = 0.01
            """上句，mask里是这个器官的掩膜，属于这个器官的地方就是1，否则就是0，
            但是我不希望mask_for_GAN[:,:,class_id]里有地方是0（后面放到GAN里面可能出问题，所以让他最小是0.01吧？）
            """
            mask_for_GAN[:, :, 0] = mask_for_GAN[:, :, 0] - full_mask * score  # 不确定？？？背景的概率得减掉啊。然后最后肯定得把减得太多的统一弄成0.01或者什么的。
            """上面似乎能做，但是很明显有缺陷的。
            1、如果有两个掩膜是重合的，怎么办？事实上很幸运地，拿来的这张图，似乎就有地方是两个掩膜重合的。
                忽然觉得这个未必是坏事儿啊，比如说这张图，在第334行、284列下方一大片位置，mask_for_GAN[:, :, 1]和mask_for_GAN[:, :, 2]都不是0.01，
                这说明这个地方有两个提出重合了，一个认为是第1类，信心得分是0.999，另一个认为是第2类，信心得分是0.640，
                然后，这样的logits被输入到D网络中，这样D网络应该很容易就判断出来，这个是假图，
                    因为真图的话，在某一个位置，mask_for_GAN[:, :, 0~6]中顶多有1个是比较大的，不太可能两个都大的啊。
            2、同样是上面的问题，mask_for_GAN[:, :, 0]一次一次地减，减得太多了
                这个也可以把减了太多的都给变成0.01吧。。不过目前倒是觉得，不变也可以，反正不是0就行吧，因为原来那个logits也有负数啊。。。
            3、为什么还会出来0？明明初始化为0.01的啊？因为被赋值为full_mask * score了啊，而full_mask里别的地方就是0啊。
                这个已经解决了，容易。
            4、前面的问题，如果module.unmold_mask里面报错，就悲剧了啊。是不是得弄个try函数啊？
                这个是一定会出问题的。。原来是测试的时候用这个东西，有一张图错了还可以不管他，训练的时候万一到了错的那张图，就跪了。。。
            """
        # print('准备返回回去了。各变量的形状：', boxes.shape, class_ids.shape, scores.shape, full_masks.shape)
        inp1_pos = np.where(detections[:, 4] == 1)[0]  # L4的位置。PS，也得加[0]，否则莫名其妙就多出来一个为1的维度。
        inp2_pos = np.where(detections[:, 4] == 2)[0]  # L5的位置。
        inp3_pos = np.where(detections[:, 4] == 3)[0]  # S1的位置。
        inp1 = detections[inp1_pos, :4]  # 所有被认为是L4（第1个类别）的提出的外接矩形。正确情况，应该是1*4的矩阵，但也有可能是2*4或者空的。下同。
        inp2 = detections[inp2_pos, :4]  # 所有被认为是L5（第2个类别）的提出的外接矩形。
        inp3 = detections[inp3_pos, :4]  # 所有被认为是S1（第3个类别）的提出的外接矩形。
        box_L4_outer = judge_empty_for_tf(inp1)  # L4（第1个类别）的“总的”外接矩形。正确情况，就是输入；不正确情况，如果多了就返回外包络，如果没有就返回全0。
        box_L5_outer = judge_empty_for_tf(inp2)
        box_S1_outer = judge_empty_for_tf(inp3)
        box_L4_outer = utils.denorm_boxes(box_L4_outer, original_image_shape[:2])
        box_L5_outer = utils.denorm_boxes(box_L5_outer, original_image_shape[:2])
        box_S1_outer = utils.denorm_boxes(box_S1_outer, original_image_shape[:2])
        box_L4_outer = box_L4_outer.astype(np.int32)
        box_L5_outer = box_L5_outer.astype(np.int32)
        box_S1_outer = box_S1_outer.astype(np.int32)
    return boxes, class_ids, scores, full_masks, mask_for_GAN, box_L4_outer, box_L5_outer, box_S1_outer