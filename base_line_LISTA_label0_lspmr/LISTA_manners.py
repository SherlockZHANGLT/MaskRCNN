import tensorflow as tf
import numpy as np
from skimage.transform import radon
import shrinkage

def get_sparse_gt(h, w, gt_boxes, projection_num):
    row1 = gt_boxes[:, 0]
    row2 = gt_boxes[:, 2]
    col1 = gt_boxes[:, 1]
    col2 = gt_boxes[:, 3]
    sparse_boxes = np.zeros([h, w])  # 图像长宽，到时候改掉。
    for i in range(len(row1)):
        r1 = np.maximum(row1[i], 0)
        r2 = np.minimum(row2[i], h-1) # 防止越界，因为那个sparse_boxes是从0~511的。有点奇怪的是，用原来数据集训练的时候，并没发生过这个问题。
        c1 = np.maximum(col1[i], 0)
        c2 = np.minimum(col2[i], w-1)
        sparse_boxes[r1, c1] = 1
        sparse_boxes[r2, c2] = 1  # 现在只是把左上右下两个角点给投影了，而不是四个。。。所以说，0°的投影轴对应的稀疏编码，应该是有2个非零值。
    theta = np.linspace(0., 180., num=projection_num, endpoint=False)
    sparse_radon = radon(sparse_boxes, theta=theta, circle=True)  # shape是(原图大小, projection_num)，就是在projection_num条直线上的投影结果。然后看了一下，第0条直线，就是投影角度为0的那个，就是原来是1的地方投影为1，而其他的投影直线，一时不知道是怎么搞出来的，可以先了解一下他确实是把金标准往不同直线做了投影，然后再看一下那个radon函数，是怎么确定的投影直线。
    sparse_radon_max = np.max(sparse_radon)  # 试着归一化一下，就是让所有的数都不超过1
    sparse_radon /= sparse_radon_max
    return sparse_radon, sparse_boxes

def LISTA_network(input_y, layer_num, init_lam, D):
    """输入：input_y是原来的图像（特征），张量，
    layer_num是LISTA层数，一个数，
    init_lam是初始化的那个收缩值，一个数
    D是字典，张量。
    输出：x（最后的那个x）就是最终的稀疏表示。印象中，048中是每一层的稀疏表示都要用来算损失，我就不要这样了吧。
    跟这个函数有关系的几个参数是：
    <tf.Variable 'mask_rcnn_model/LISTA_dict_learning/weights:0' shape=(1024, 2048) dtype=float32_ref>,
        --这个是字典，1024是输入y的长度，2048是4*512即投影数*图像长宽
    <tf.Variable 'mask_rcnn_model/LISTA_dict_learning/S:0' shape=(2048, 2048) dtype=float32_ref>,
        --这个是那个S矩阵
    <tf.Variable 'mask_rcnn_model/LISTA_dict_learning/lam_0:0' shape=() dtype=float32_ref>,
    <tf.Variable 'mask_rcnn_model/LISTA_dict_learning/lam_1:0' shape=() dtype=float32_ref>,
    <tf.Variable 'mask_rcnn_model/LISTA_dict_learning/lam_2:0' shape=() dtype=float32_ref>,
        --这3个（layer_num个）是那layer_num个lam参数。
    """
    layers = []
    M, N = D.get_shape().as_list()
    B = tf.transpose(D)
    I_N = tf.eye(N, dtype=tf.float32)
    S = tf.Variable(I_N - tf.matmul(B, D), dtype=tf.float32, name='S')
    By = tf.matmul(B, tf.transpose(input_y))  # 048文章26式的那个\bold{B}\bold{y}，'mask_rcnn_model/LISTA_dict_learning/MatMul_1:0'
    # 上句，注意到他们的程序里，y是(特征数, 批大小*num_rois)的，所以这里转置一下。
    layers.append(('Linear', By, None))  # 这儿是append了一个tuple啊，然后这个tuple里有一个By_。
    eta = shrinkage.simple_soft_threshold  # eta就是收缩函数。
    lam0 = tf.Variable(init_lam, name='lam_0')
    x = eta(By, lam0)  # 第0层LISTA，'mask_rcnn_model/LISTA_dict_learning/mul:0' shape=(2048, ?)
    layers.append(('LISTA T=1', x, (lam0,)))
    for t in range(1, layer_num):  # 加入layer_num个层，每个层都是048中的图3所示。
        lam = tf.Variable(init_lam, name='lam_{0}'.format(t))
        x = eta(tf.matmul(S,x) + By, lam)  # 048文章26式，这次加了\bold{S}\hat{\bold{x}_t}。
        layers.append(('LISTA T='+str(t+1), x, (lam,)))  # 每一层的xhat_都给append进去。
    x = tf.transpose(x)  # 稀疏表示，再给他转回去。现在的shape应该是(批大小*每张图中物体数, 投影数*图像长宽)，即(2*num_rois, 投影数*512)的。
    # x是<tf.Tensor 'mask_rcnn_model/LISTA_dict_learning/transpose_2:0' shape=(?, 2048) dtype=float32>
    return layers, x  # 这个x是最后的x