import tensorflow as tf
import numpy as np

def sparse_func(data, dictionary, k_value):
    layer = tf.matmul(data, dictionary)  #
    topk_value = tf.nn.top_k(layer, k_value)
    topk_masks = tf.cast(tf.greater_equal(layer, tf.reduce_min(topk_value.values)), tf.float32)
    sparse = layer * topk_masks
    """那个tf.nn.top_k是对每一行取的，所以不需要用try5里的那个for循环了，傻掉了。
    """
    reconstructed_layer = tf.matmul(sparse, tf.transpose(dictionary))
    return sparse, reconstructed_layer

def get_label_consist_paras(input_shared_features, target_class_ids_reshaped, compress_size):
    """这个函数应该是融入到tf的计算图中的。
    所以，输入的特征就是那个shared，是(batch_size*TRAIN_ROIS_PER_IMAGE, 1024)的 -- 对应原来是(batch_size, 784)的；
    然后那个D矩阵，应该是(1024, compress_size)的 -- 对应原来是[784, compress_size]的；
    那个T矩阵，应该是(compress_size, compress_size)的；
    那个Q矩阵，应该是(batch_size*TRAIN_ROIS_PER_IMAGE, compress_size)的 -- 对应原来的[batch_size,compress_size]；
        不过，他有一个整个的Q矩阵，然后每次从这个整个的Q矩阵中选取几个，这个怎么弄呢？？能不能每一次就生成这一次的Q矩阵就可以了？
    还得输入那个标签，就是每个输入特征对应的标签，应该就是那个target_class_ids再给reshape成(batch_size*TRAIN_ROIS_PER_IMAGE, )的，
        这个是一个数的标签，不过好像也不用变成1热的吧，因为那个弄Q的循环里用的就是一个数的标签。

    现在，既然发现D和T和W矩阵都没必要用KSVD去初始化，那么，这个东西只需要构造字典的标签，然后弄出来那个一致性矩阵Q就可以了。
    而且，应该是只需要弄这一个批次的一致性矩阵，而没必要弄所有的批次的。
    """
    numClass = 10 # config.NUM_CLASSES
    assert compress_size % numClass == 0, '类别数必须能够被字典列数整除。'
    numPerClass = int(compress_size / float(numClass))  # 必须给变成int，否则放到tf.py_func的时候会闹麻烦。
    _dictLabel = np.zeros((numClass, numClass * numPerClass))  # 初始化为一大堆np的0。
    for classid in range(numClass):
        _dictLabel[classid, numPerClass * classid:numPerClass * (classid + 1)] = 1.
    # 上面的东西应该都是可以用np做的，后面要用tf了。
    sample_num = input_shared_features.shape[0]  # 应该是batch_size*TRAIN_ROIS_PER_IMAGE
    Q = np.zeros((compress_size, sample_num), dtype=np.float32)  # 弄成float32类型的，和get_label_consist_paras_tf函数里输出的tf.float32对应上。
    for frameid in range(sample_num):  # 这个是对每个样本循环的
        maxid1 = target_class_ids_reshaped[frameid]
        for itemid in range(compress_size):
            label_atom = _dictLabel[:, itemid]
            maxid2 = np.argmax(label_atom)
            if (maxid1 == maxid2):
                Q[itemid, frameid] = 1
            else:
                Q[itemid, frameid] = 0
    Q = np.transpose(Q)  # 转置一下，这儿和原来那个程序有点不一样。那个是在外面才转置的。
    return Q

def get_label_consist_paras_tf(input_shared_features_tf, target_class_ids_reshaped_tf, compress_size):
    Q_tf = tf.py_func(get_label_consist_paras, [input_shared_features_tf, target_class_ids_reshaped_tf, compress_size],
                   tf.float32)  # 后面那个是输出的类型。如果只有一个输出，就不应该要[]了。
    return Q_tf

def get_S_mat_for_in_plane_loss(lab_for_dict):
    """
    lab_for_dict是(batch_size*TRAIN_ROIS_PER_IMAGE, )的，和上面那个target_class_ids_reshaped一样。
    输出是044文章的S矩阵（这里是分类问题，就只有0和1了吧）
    """
    sample_num = lab_for_dict.shape[0]
    S = np.zeros((sample_num, sample_num), dtype=np.float32)
    for i in range(sample_num):
        for j in range(i+1, sample_num):  # i+1是因为对角线上的不需要置1。
            if lab_for_dict[i] == lab_for_dict[j]:
                S[i, j] = 1
                S[j, i] = 1
    D = np.zeros((sample_num, sample_num), dtype=np.float32)
    for i in range(sample_num):
        D[i, i] = np.sum(S[i, :])  # 就是老庞文章的那个D矩阵，对角阵。
    L = D - S
    return S, D, L

def get_S_mat_for_in_plane_loss_tf(lab_for_dict_tf):
    S_tf, D_tf, L_tf = tf.py_func(get_S_mat_for_in_plane_loss, [lab_for_dict_tf], [tf.float32, tf.float32, tf.float32])
    # 后面那个是输出的类型。
    return S_tf, D_tf, L_tf

def main(_):  # 检查用的，这个py文件作为函数执行的时候不用执行这个。
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('E:/TensorFlow/1_first_codes/Mnist_data/', one_hot=True)  # 大电脑
    # mnist = input_data.read_data_sets('A:/pycharm_projects/2_mnist_recog/mnist-master/Mnist_data/')  # 小电脑
    # 上面，那个one_hot=False好像就是一个数的标签了。
    batch = 2 * 20  # 模拟那个batch_size*TRAIN_ROIS_PER_IMAGE
    fake_data = False
    images_feed, labels_1hot = mnist.train.next_batch(batch, fake_data, shuffle=False)  # 好像得到的是一个数的标签吧。
    # images_feed应该是(批大小, 784)的，labels_feed应该是(批大小,)的一个数的标签。
    labels_feed = np.argmax(labels_1hot, axis=1)
    Q = get_label_consist_paras(images_feed, labels_feed, compress_size=400)  # (400(compress_size), 40(样本数/批大小))
    print(Q)
    S, D, L = get_S_mat_for_in_plane_loss(labels_feed)
    print(S)
    print('直接用py的程序执行完毕，下面试试用tf的py_func。')

    input_feats_tf = tf.placeholder(tf.float32, shape=[batch, 784], name="feats")
    input_labels_tf = tf.placeholder(tf.float32, shape=[batch, ], name="label")
    Q_tf = get_label_consist_paras_tf(input_feats_tf, input_labels_tf, compress_size=400)
    S_tf, D_tf, L_tf = get_S_mat_for_in_plane_loss_tf(input_labels_tf)
    feed_dict = {input_feats_tf: images_feed, input_labels_tf: labels_feed}
    sess = tf.Session()  # 现在不需要初始化，因为反正也没有变量。。
    _Q_tf = sess.run(Q_tf, feed_dict=feed_dict)
    _S_tf, _D_tf, _L_tf = sess.run([S_tf, D_tf, L_tf], feed_dict=feed_dict)
    print(_Q_tf)
    print(_S_tf)
    print('结束。')

if __name__ == '__main__':
    tf.app.run()