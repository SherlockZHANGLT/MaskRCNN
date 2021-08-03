import tensorflow as tf
import keras.backend as K

def get_calibrated_CPV(in_CPV, feature_params, transition_params):
    """输入的in_CPV应该是补了零的，大小是(batch_size, num_classes-1, num_classes)，
        第0个是批大小，第1个是一共顶多是num_classes-1个椎骨，第2个是一共有num_classes种类别。
    """
    CPV_modified = []
    for i in range(in_CPV.shape[0]):  # 就是每次弄一个样本了，然后训练完了之后再给他append到一起吧。
        belief = beliefs(in_CPV[i, :, :], feature_params, transition_params)  # 这个地方不能去掉补零，否则后面chain_potentials函数会报错。
        pairwise_p = pairwise_prob(belief)  #  <tf.Tensor 'Exp:0' shape=(8, 10, 10) dtype=float32>
        cpv_modified = single_prob(pairwise_p)  # <tf.Tensor 'concat_3:0' shape=(9, 10) dtype=float32>
        CPV_modified.append(cpv_modified)
    CPV_modified = tf.stack(CPV_modified, axis=0)  # 要求输入的两张图节点数一样，现在都是num_classes-1个（含补零）
    return CPV_modified

def beliefs(in_CPV_this, feature_params, transition_params):
    """
    Returns a numpy array of size (w-1) x k x k
    返回一个numpy数组（就是信仰值）
    输入:
        in_CPV_this: 某个椎骨的CPV，shape是(num_classes-1, num_classes)。第一次循环是<tf.Tensor 'strided_slice_18:0' shape=(9, 10) dtype=float32>
        feature_params：特征参数矩阵，shape是(10(类别数), 10(特征个数，因为现在输入的是CPV，所以特征个数也是10))。
        transition_params：转移参数矩阵，shape是(10(类别数), 10(类别数))。
    """
    psi = chain_potentials(in_CPV_this, feature_params, transition_params)  # <tf.Tensor 'stack:0' shape=(8, 10, 10) dtype=float32>
    delta_bwd, delta_fwd = message_passing(psi)  # 第一次循环是<tf.Tensor 'stack_1:0' shape=(7, 10) dtype=float32>和<tf.Tensor 'stack_2:0' shape=(7, 10) dtype=float32>
    k = delta_fwd.shape[1]  # 10，就是10种标签。
    delta_fwd = tf.concat(([tf.zeros(k)], delta_fwd), axis=0)
    delta_bwd = tf.concat((delta_bwd[::-1], [tf.zeros(k)]), axis=0)
    belief = psi + delta_fwd[:, :, tf.newaxis] + delta_bwd[:, tf.newaxis, :]  # 第一次循环是<tf.Tensor 'add_40:0' shape=(8, 10, 10) dtype=float32>
    return belief

def chain_potentials(in_CPV_this, feature_params, transition_params):
    """
    Computes the clique potentials of the entire chain.
    Returns a (w-1) x C x C numpy array
    计算整个链的势能，返回一个numpy数组（就是链的势能）。见本函数最后的总结。
    输入:
        in_CPV: 就是输入的CPV，shape是(num_classes-1, num_classes)。相当于是原来那个程序里的phi了，这儿就是那个MRCNN的类别分支输出的_mrcnn_class
        transition_params：转移参数矩阵，shape是(num_classes, num_classes)。
    """
    phi = feature_potentials(in_CPV_this, feature_params)  # 那些节点的初始CPV，应该就对应于掩膜RCNN里的预识别的CPV。
    # transitions = [(node, None) for node in phi[:-2, :]] + [(phi[-2, :], phi[-1, :])]  # 这个不行，得用下面那个，效果似乎是一样的。
    transitions = []
    for i in range(phi.shape[0] - 2):
        transitions.append((phi[i,:], None))
    transitions = transitions + [(phi[-2, :], phi[-1, :])]  # 有点烦的是，tf的在这儿不太好观察。。
    psi = [transition_potential(node1, node2, transition_params) for node1, node2 in transitions]
    psi = tf.stack(psi, axis=0)  # 希望这个时候是个(节点数, 类别数, 类别数)的东西。 <tf.Tensor 'stack:0' shape=(8, 10, 10) dtype=float32>
    return psi  #

def feature_potentials(word, feature_params):
    """
    feature_params, C x n numpy array of feature parameter.
    word ,a w x n numpy array of feature vectors,
    word= X_test[i]
    计算特征势能。
    其实就是：图片格式的单词（8(单词长度)*321的矩阵）点乘上特征参数矩阵（10*321的矩阵）的转置，其中321 10的意义见主函数。
        输出的“特征势能”就是8(单词长度)*10的矩阵，相当于是每个字母（即信息传递中所谓的节点）的CPV初始值。
    这一步就应该是相当于那些CNN网络的极度化简版。本来以为应该不需要的，但发现如果不要这个，结果就不怎么好，有点奇怪。。。
    """
    phi = tf.matmul(word, tf.transpose(feature_params))
    return phi

def transition_potential(feat_pot1, feat_pot2, transition_params):
    """
    Absorb node potential into a pairwise potential, for positions (t, t+1).
    output: (log) pairwise potential function.  (a table, e.g. array)
    Returns a C x C numpy array, where k is the size of the alphabet.
    对于位置t和t+1，把节点势能弄成一个成对势能。返回一个C*C（10*10）矩阵，C是num_classes（10）。
        输入的feat_pot1是第t个位置的节点势能（即第t个字母是那10个标签的打分值）
        输入的feat_pot2是第t+1个位置的节点势能（即第t+1个字母是那10个标签的打分值），也有可能是None
        输入的transition_params是转移参数矩阵，里面第i行第j列，就表示第t个字母的标签是i而第t+1个字母的标签是j的打分值。
    返回的tran_pot（成对势能）的每个元素，应该是代表输入的两个字母（对应feat_pot1和2）分别取某个标签的logits，详见下面。
    """
    # print (feat_pot1[:, np.newaxis].shape, transition_params.shape)
    tran_pot = transition_params + feat_pot1[:, tf.newaxis]  #
    # 上句，tran_pot（转移势能）是转移矩阵（10*10），每一列都加上feat_pot1的转置（特征势能phi矩阵的一行，10*1向量，
    #     转置成10行1列的向量）即，在“某个字母为某个标签且下一个字母是另一个标签”的得分上，
    #     加上了“单词中的某个字母是某个标签”的得分。具体例子见下面。
    if feat_pot2 is not None:
        tran_pot += feat_pot2  # 如果feat_pot2不是None，那么再加上这个feat_pot2。也是10*10矩阵+10*1向量。
        # 这一次，是按行加的，即tran_pot是10*10的矩阵，feat_pot2是10*1（1行10列）向量，
        #     然后tran_pot中的每一行都和feat_pot2相加，得到新的feat_pot2矩阵。
    return tran_pot

def message_passing(psi):
    """
    Message passing algorithm
    input: (log-) potential
    outputs: forward/backward messages
    【重要】信息传递算法。输入是势能psi（这4个字母两两之间的“联合logits”、即“成对势能”，详见上面，
        已经含有“特征参数矩阵”和“转移参数矩阵”的信息了），输出的向前和向后的信息。
    """
    # Backward messages
    back = []
    prev_msgs = tf.zeros(psi.shape[1])  # 似乎是previous message，即以前的信息。
    # for pairs in psi[:0:-1, :, :]:
    #     message = tf.reduce_logsumexp(pairs + prev_msgs, axis=1)
    #     back.append(message)
    #     prev_msgs += message
    psi_remove_0_inverse = psi[:0:-1, :, :]
    for i in range(psi_remove_0_inverse.shape[0]):
        message = tf.reduce_logsumexp(psi_remove_0_inverse[i, :, :] + prev_msgs, axis=1)
        back.append(message)
        prev_msgs += message

    # Forward messages
    fwd = []
    prev_msgs = tf.zeros(psi.shape[1])
    # for pairs in psi[:-1]:
    #     # 这一次，pairs是第一次循环的时候，就是psi(0,:,:)；第二次循环的时候，就是psi(1,:,:)，相当于是正向传递信息。
    #     message = tf.reduce_logsumexp(pairs + prev_msgs[:, tf.newaxis], axis=0)
    #     fwd.append(message)
    #     prev_msgs += message
    psi_remove_last = psi[:-1, :, :]
    for i in range(psi_remove_last.shape[0]):
        message = tf.reduce_logsumexp(psi_remove_last[i, :, :] + prev_msgs[:, tf.newaxis], axis=0)
        fwd.append(message)
        prev_msgs += message

    back = tf.stack(back, axis=0)  # <tf.Tensor 'stack_1:0' shape=(7, 10) dtype=float32>，那个for循环每次执行，得到的信息都在这儿。
    fwd = tf.stack(fwd, axis=0)  # <tf.Tensor 'stack_2:0' shape=(7, 10) dtype=float32>
    return (back, fwd)

def pairwise_prob(belief):
    """
    pairwise marginal probabilities.
    这一步就是软最大的过程。
    """
    pairwise_prob_value = tf.exp(belief - tf.reduce_logsumexp(belief, axis=(1, 2))[:, tf.newaxis, tf.newaxis])
    return pairwise_prob_value

def single_prob(pairwise_p):
    """
    singleton marginal probabilities.
    单个字母的边缘概率。输入pairwise_p是软最大后的信仰值，即软最大后的路径得分（belief详见beliefs函数）
    边缘概率是对联合分布的某一个变量的分布求和了，这个被边缘化（求和）的变量是哪个，就是通过“按行求和”还是“按列求和”决定的。
    """
    a = tf.reduce_sum(pairwise_p, axis=2)  # <tf.Tensor 'Sum:0' shape=(8, 10) dtype=float32>
    b = tf.reduce_sum(pairwise_p[-1], axis=0)  # <tf.Tensor 'Sum_1:0' shape=(10,) dtype=float32>
    cpv_modified=tf.concat((a, b[tf.newaxis, :]), axis=0)
    return cpv_modified

def path_score_func(inputs, labels, Psi_mat):
    """计算目标路径的相对概率（还没有归一化），这个是5式的分子（对数）。
    要点：逐标签得分，加上转移概率得分。
    技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        上句的技巧，输入的东西是y_pred和y_true，也就是说用预测值和金标准算路径得分，和18中的那个np程序不一样，
        因为这个时候知道了金标准，所以就直接取了正确的“某个汉字取某个标签的单个得分”加上
            “某两个汉字取某两个标签的路径得分”（详见下面注释），而18中的那个np的程序是不知道金标准的，所以用动态规划计算了所有的可能组合的得分。
        具体地，那个labels是(128,3,4,4)的张量，128是批次的大小，然后每个单词都有一个(3,4,4)的张量，
            这个3就对应单词的长度（因为长度是4啊），然后4*4的就是每两个汉字之间的标签的“路径得分”。
            这个4*4的矩阵就对应18里的那个转移参数矩阵，然后因为有3个汉字，所以有3个矩阵，有点像是那个“psi”张量，
            只不过这儿还没加那个“单个得分”呢。
        然后，现在这个3*4*4的张量不是1就是0（因为是用金标准弄的），只有labels1和labels2都是1的地方，才是1，否则就是0，
            所以，似乎就是两个相邻汉字标签的路径金标准为1（即相邻的汉字确实是这两个标签）的时候，才把这个labels记作1，
            那么就用trans*labels把trans的相应位置的标签保留，这就是所谓的转移得分。
    """
    point_score_mat = inputs*labels
    point_score = K.sum(K.sum(point_score_mat, 2), 1, keepdims=True) # 逐标签得分
    # 上句，inputs*labels（用“预测”点乘“目标”）完了是shape=(?, ?, 4)，然后对后面两维度加和，
    #     第二次加和的时候保留维度为shape=(?, 1)（而不是shape=(?,)）。
    # 【重要】这个似乎是《说明-CRF的原理》中的7式中跟h有关的项的求和，也就是单个标签的得分。
    #     现在他用金标准的标签去乘了，这个原因是：原来预测的y_pred（即输入inputs）是每个汉字取所有标签的概率，
    #         和labels相乘后，就只保留了预测对的那几个标签的概率（比如说第0个汉字金标准标签是1，
    #         这儿就把第0个字母被预测为2、3、4的概率都删了，只留下预测为1的概率），然后加和，
    #         就相当于是所有正确单个标签的得分和，也就是7式中所有的h(y1;x)...h(yn;x)求和。
    labels1 = K.expand_dims(labels[:, :-1], 3)
    # 上句，本来labels应该是(?,?,?)的（predict掉后是(128,4,5)），然后在第1个维度取前面若干个（最后一个不取），
    #     就得到(128,3,5)的，然后在第3个维度扩展，就成了shape=(?, ?, ?, 1)也就是(128, 3, 4, 1)
    labels2 = K.expand_dims(labels[:, 1:], 2)
    # 类似上上句，只不过这次是第1个维度取了后面若干个（第0个不取），上句的输出shape=(?, ?, 1, ?)即(128, 3, 1, 4)
    labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
    # 上句，shape=(?, ?, ?, ?)，predict掉后是(128,3,4,4)。为什么是用lables，为什么是乘，我怎么感觉是加似的？
    #     而且还一个问题，这儿用的是金标准相乘啊，这一乘应该是大多数都乘成0了吧。。
    trans = K.expand_dims(K.expand_dims(Psi_mat, 0), 0)  # shape=(1, 1, 4, 4)
    # 上句，本来self.trans是4*4的，给扩展了两个维度。
    trans_score_mat = trans*labels  # predict掉后是(128,3,4,4)，似乎和那个信仰值矩阵有点像。
    trans_score = K.sum(K.sum(trans_score_mat, [2,3]), 1, keepdims=True)  # shape=(?, 1)
    # 【重要】上句是《说明-CRF的原理》中的7式中跟g有关的项的求和，和point_score的计算有点像。
    #     这儿是先用labels1 * labels2得到了前后两个标签分别取某两个值的“联合标签”，然后去乘以trans，
    #     也就是说，只计算预测对的那几个路径的转移得分。（比如说第0个汉字金标准标签是1，第1个的金标准标签也是1
    #         这儿就只保留了第0个字母为1且第1个字母为1的转移矩阵中的参数，而其他的都删了）然后加和，
    #         就相当于是所有正确转移得分之和，也就是7式中所有的g(y1;y2)...g(yn-1;yn)求和。
    # 和18里的程序不一样的是，他似乎是用金标准先在各个路径里选了金标准标签对应的路径，而18则是计算了所有路径。
    path_scores = point_score+trans_score # 两部分得分之和
    return path_scores, point_score_mat, point_score, labels1, labels2, labels, trans_score_mat, trans_score

def path_loss(y_true, y_pred, Psi_mat_valid, ignore_first_label): # 目标y_pred需要是one hot形式
    """
    y_pred就是主函数传进来的的那个tag_score，<tf.Tensor 'crf_1/Identity:0' shape=(?, ?, 5) dtype=float32>
        这个可以用keras查看，predict掉后，shape是(128, 4, 5)，即批大小、最长的词长度（见那个train_generator函数）、标签种类数。
    y_true是<tf.Tensor 'crf_1_target:0' shape=(?, ?, ?) dtype=float32> 最后的?应该也是5。
    """

    def log_norm_step(inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        输入inputs的shape是(?, 4)，应该是i时刻的变量值；states的shape也是(?, 4)；输出就是i+1时刻的变量值。
        【重要】看到K.rnn的说明，说“step_function的inputs是tensor with shape `(samples, ...)` (no time dimension),
            representing input for the batch of samples at a certain time step.”，这似乎就是说，这个log_norm_step函数
            （即K.rnn的step_function）的输入，应该就是某个时刻点的一个小批量的样本。所以，如果我想测试这个函数的话，那么，
            输入的inputs大概是y_pred[:,1,:]，即第1个时刻的标签打分；然后states大概是init_states，即第0时刻的状态。
        """
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1) -- shape=(?, 9, 1)
        trans = K.expand_dims(Psi_mat_valid, 0)  # (1, output_dim, output_dim) -- shape=(1, 9, 9)
        output = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)  -- 见下
        # 上句，states+trans的shape=(?, 4, 4)（说明这个加法也是先把states弄成了(?, 4, 4)的，然后把trans也变成了(?, 4, 4)的），
        #     最后输出output的shape=(?, 4)。然后这个output就是时刻t的变量值（第t个汉字的标签）和时刻t-1的变量值（第t-1个汉字的标签）之间的关系，
        #     【重要】所以他最后return的是output+inputs，是一个累积得分（输入是t时刻之前的累积得分，输出是在输入上加了这一时刻的取不同标签的得分，
        #         也就是t时刻之后的累积得分），这应该就对应另一个程序里的
        #         “第i个字母取标签a的‘单个得分’”（这儿的inputs） +“第i个字母取标签a而第i+1个字母取标签b的‘路径得分’”（这儿的outputs）。
        #     然后这儿多了个batch（第0维的?），不过不妨碍，因为无论有没有这一维度，都是第i+1个汉字的标签=第i个汉字的标签单个得分+路径得分，
        #     这里把打分作为“不同时刻的变量取值”，即后面【基础K.rnn】中的向量序列(y1,y2,…,yT)。
        return output + inputs, [output + inputs]
        # 好像作为递归函数的，都要返回这么两个东西。见https://www.jiqizhixin.com/articles/2018-06-29-3。
        # 然后似乎只能放在这个path_loss函数里面，因为K.rnn调用本函数的时候，只给了inputs, states这两个输入，第三个输入(Psi_mat)没法输进来，
        # 就只能放在这个外层函数里面，并且把这个函数输入的Psi_mat删掉，这样Psi_mat可以作为全局变量弄进来。

    mask = 1-y_true[:,1:,0] if ignore_first_label else None
    y_true,y_pred = y_true[:,:,1:],y_pred[:,:,1:]
    # 上句执行后，y_pred的shape变成了(?, ?, 4)，即最后一个标签扔掉了；y_true的仍然是(?, ?, ?)，但最后的那个5应该也变成了4
    init_states = [y_pred[:,0,:]] # 初始状态，shape=(?, 4)。
    # y_pred[:,0]可能就是y_pred[:,0,:]，形状是对的。那就相当于取了这个批次中所有词语的、第0个字的、预测标签。然后这个第0个字就是所谓的时间维度。
    log_norm,_,_ = K.rnn(log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）  shape=(?, 4)
    log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）  shape=(?, 1)
    path_score, _, _, _, _, _, _, _ = path_score_func(y_pred, y_true, Psi_mat_valid) # 计算分子（对数）
    loss = log_norm - path_score # 即log(分子/分母)
    return loss

def max_in_dict(d): # 定义一个求字典中最大值的函数
    # key,value = d.items()[0]   原来的，好像是Python2的
    key,value = list(d.items())[0]
    # for i,j in d.items()[1:]:  原来的，好像是Python2的
    for i,j in list(d.items())[1:]:
        if j > value:
            key,value = i,j
    return key,value

def viterbi(nodes, trans): # viterbi算法，跟前面的HMM一致
    """输入nodes应该是个字典，就是一个单词（一张图）中每个字母（每个探测结果）取不同标签的logits。trans就是转移矩阵。"""
    paths = nodes[0] # 初始化起始路径
    # 上句，paths是{'BG': 9.341016e-05, 'L4': 0.9998233, 'L5': 6.525709e-05, 'S1': 1.799106e-05}，
    # 即，所谓起始路径就是第0个节点（第0个探测结果）取各个标签的单个得分（logits）。
    for l in range(1, len(nodes)): # 遍历后面的节点
    # 上句，每循环一次，就弄进来1个节点（1个汉字、或者1个探测结果）。l是从1到2（因为删了补零之后，一共是3个节点、即3个探测结果，
    # 其中第0个已经在初始化的时候放在paths里了，所以这个循环就处理第1和2两个）。
        paths_old,paths = paths,{}
        # 上句，老路径被更新为当前的路径，当前路径置为空。似乎有点像“信息传递”，（本质上就是单个打分+转移得分）。
        # 第一次对l的循环（l=1的时候），paths_old就是刚才的paths，也就是“第0个节点取各个标签”的得分，paths就是空的。
        # 第二次对l的循环（l=2的时候），paths_old就是刚才的paths，这次就是“第1个节点取各个标签，且第0个节点取‘使得第1个节点取相应标签的路径总得分最大’的标签”的得分。
        for n,ns in nodes[l].items(): # 当前时刻的所有节点。
        # 上句的for循环，每循环一次，读入一个节点，也就是一个探测结果取各个标签的得分。
        # 具体地，nodes[l]是第l个节点（探测结果）可以取的标签-得分组成的dict；然后n就是标签（如'L4'），ns就是对应的得分。
        # 然后这个循环是遍历了第l个节点所有可能的取值和得分，比如说，第一次循环l=1的时候，
        #     就是n就是BG、L4、L5、S1这四个值，然后ns就是第1个结果取这四个标签的得分。这种用dict的方法值得学习一下。
        # 第二次循环l=2的时候，n还是BG、L4、L5、S1这四个值，然后ns就是第2个结果取这四个标签的得分，和l=1时候的物理意义一样。
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items():  # 截止至前一时刻的最优路径集合。
                # 类似于上面对n/ns的循环，这个循环是遍历了第l个节点之前的路径（paths_old）的得分，
                # 比如说，第一次对l的循环（l=1的时候），就是在固定某个n/ns取值（即第l=1个节点的标签和得分），然后遍历p（第l=1个节点之前的路径）
                #     的BG、L4、L5、S1这四个值，然后ps就是第0个结果（因为第l=1个节点之前的路径只有第0个结果）取这四个标签的得分。
                # 【重要】也就是说，对n/ns和p/ps的这两个for循环，就已经遍历了所有的已有路径得分、当前节点单个得分、当前节点与前面路径的转移得分，
                # 比如说，l=1的时候，ns就遍历了第1个探测结果是BG、L4、L5、S1这四个值的得分，
                #    ps则遍历了第0个结果是BG、L4、L5、S1这四个值，然后trans[p[-2:]+n]则是第0和1两个结果取这4*4=16种可能性的转移得分，
                #    并且在下句把它们都加起来，这样每次计算score就得到这16种可能性中某一种的“路径总得分”。
                # l=2的时候，ns就遍历了第2个探测结果是BG、L4、L5、S1这四个值的得分，
                #    ps则遍历了第0和1个结果是L4L4、L4L5、L5S1、S1BG这四种组合情况（因为这是“第1个节点取各个标签，
                #         且第0个节点取‘使得第1个节点取相应标签的路径总得分最大’的标签”的情况），
                #     然后trans[p[-2:]+n]则是“第2个结果”和“第0-1个结果的组合”取这4*4=16种可能性的转移得分，
                #     并且在下句把它们都加起来，这样每次计算score就得到这16种可能性中某一种的“路径总得分”。
                score = ns + ps + trans[p[-3:]+n] # 计算新分数...p[-1:]是因为要取上一个标签，且标签有1个字符。
                # 上句中的得分，是当前节点和老路径的、任意一种标签组合的“路径总得分”，见上面注释。
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path,max_score = p+n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径。
            # 上句是处理第l个探测结果的时候，固定n/ns（这个结果的某一种标签及其得分），然后选出来最大的p和ps（使得这个结果取这个标签的“路径总得分”最大的标签）。
            # 比如说，l=1的时候，首先看n=BG（第1个结果是BG类）、ns是第1个结果为BG类的得分，
            #     然后在这种情况（第1个结果是BG类的情况）下，遍历第0个结果的所有取值，把最大的存起来，
            #     比如说，如果第1个结果是BG类，上面for循环遍历所有p/ps后发现第0个结果是S1类的时候，这个得分最大，
            #         所以paths就变成了{'S1BG': 1.8879622949950026}；
            #     然后n/ns循环，还是l=1，这时候循环到了n=L5（第1个结果是L5类）、ns是第1个结果为L5类的得分，
            #     在这种情况（第1个结果是L5类的情况）下，遍历第0个结果的所有取值，把最大的存起来，
            #     比如说，如果第1个结果是L5类，上面for循环遍历所有p/ps后发现第0个结果是L4类的时候，这个得分最大，
            #         所以paths就变成了{'L4L5': 9.095800254619148, 'S1BG': 1.8879622949950026}。
            #     以此类推，当n/ns的for循环结束（第1个结果遍历完了所有取值）之后，这个paths里应该有4个元素，分别为第1个结果取某个标签，
            #         而第0个字母取另一个标签、且使得第0-1两个字母的总路径得分最大的路径。
            #     比如，{'L4L4': -0.5307302388086061, 'L4L5': 9.095800254619148, 'L5S1': 1.799473249388182, 'S1BG': 1.8879622949950026}。
            #     这个时候，最外面对l的循环，l=1的情况就处理完了，也就是说第1个探测结果就处理完了。
            # l=2的时候，也是首先看n=BG（第2个结果是BG类）、ns是第2个结果为BG类的得分，
            #     然后在这种情况（第2个结果是BG类的情况）下，遍历第0-1个结果的四种L4L4、L4L5、L5S1、S1BG组合，把最大的存起来，
            #     比如说，如果第2个结果是BG类，上面for循环遍历所有p/ps后发现第0-1个结果的组合是L4L5的时候，这个得分最大，
            #         所以paths就变成了{'L4L5BG': 7.791094879922171}；
            #     然后n/ns循环，过程和n=1的时候差不多，只不过循环完了之后，paths里就有三个东西了，
            #     如：{'L4L5BG': 7.791094879922171, 'L4L5L4': 7.776442191681571, 'L4L5L5': 8.362591365972612, 'L4L5S1': 11.89453428769759}。
            #     这个时候，l=2也就处理完了，注意到，前面两个都是L4L5（事实上，必须是L4L4 L4L5 L5S1 S1BG这四种组合中的一个），
            #         然后他现在都取了L4L5，正是因为前面两个标签为L4L5的得分最大（9点多分），比其他三种组合都大（不到2分）。
    return max_in_dict(paths)

# ----------------以上是信息传递的核心函数，下面是一些用到的外围函数，在那个crf程序里也用到了。----------------
import numpy as np
import MRCNN_utils

def trim_zero(input):
    """处理一张图的函数，裁剪补零。"""
    ix_unpadded = np.where(np.sum(input, axis=1) > 0)[0]  # 非补0的位置
    output = input[ix_unpadded]
    return output, ix_unpadded

def trim_gt(input_gt, num_classes):
    """就是把金标准也弄成10*10的东西，先把补零都删了，然后按照坐标顺序排序。"""
    ix_unpadded = np.where(np.sum(input_gt, axis=1) > 0)[0]  # 非补0的位置
    gt = input_gt[ix_unpadded]
    sorted = gt[gt[:, 0].argsort()]  # 按照第一列（上面的y坐标）排序
    max_class_id = np.max(sorted[:, 4])
    min_class_id = np.min(sorted[:, 4])
    P_before = int(num_classes - 1 - max_class_id)
    P_after = int(min_class_id - 1)
    # padded = np.pad(sorted, ((P_before, P_after), (0, 0)), 'constant', constant_values=(0))
    padded = np.pad(sorted, ((0, P_before+P_after), (0, 0)), 'constant', constant_values=(0))
    return padded

def batch_processing(process_func, input_batch, **kwargs):
    """现在这个函数，可以处理参数函数返回一个或多个变量的情况了。
    """
    processed = []
    processed_all = []  # 这样，就不需要写一大堆的(参数名_all)这样的变量，然后一个一个append在concatenate了。
    for i in range(input_batch.shape[0]):
        slice = input_batch[i]  # 相当于是input_batch[i,:,:]或者[i,:]或者[i,:,:,:]什么的
        processed = process_func(slice, **kwargs)  # 输入这个func函数的参数，这儿不应该有，而应该通过**kwargs在外面输入
        processed_all.append(processed)
    assert type(processed) in [np.ndarray, tuple], '暂时不支持输出别的类型'
    if isinstance(processed, np.ndarray):  # 【基础】判断是不是np.array。如果是的话，就认为这个函数返回了1个变量。似乎还没考虑返回一个数的情况。。
        processed_all = np.stack(processed_all, axis=0)
        return processed_all
    else:  # 如果不是，就认为这个函数返回的是tuple，也就是多个变量。
        processed_all_zipped = list(zip(*processed_all))
        # 上句，processed_all_zipped是个list，里面的每一个元素都是tuple（这个时候好像不太方便给他变成np.array，毕竟这个函数
        #     不知道process_func的输入输出，也就不知道他有几个元素啊）。
        result = [np.stack(o, axis=0) for o in zip(processed_all_zipped)]
        # 上句，弄成list，然后主函数里可以直接用，见调用的时候。但是调用的时候就会发现，比起单个输出的，就都多了一维。不过，可以在这儿就把它删掉。
        result = [np.squeeze(r) for r in result]  # 删掉多余的维度（是前面的zip等操作搞出来的）
        if len(result) == 1:
            result = result[0]
        return result

def batch_processing_multi_input_and_output(process_func, input_batches, **kwargs):
    """现在试试能否输入也是多个张量。上一个函数里的input_batch变成了input_batches，就是说输入的东西是多个张量了。
    当然，要求每个张量的第0维（即有几个矩阵）都相等。
    """
    if not isinstance(input_batches, list):  # 先把输入的np.array变成list。
        input_batches = [input_batches]
    processed_all = []  # 这样，就不需要写一大堆的(参数名_all)这样的变量，然后一个一个append在concatenate了。
    for i in range(input_batches[0].shape[0]):
        slices = [x[i] for x in input_batches]  # 批次中的、一张图中的所有输入张量。
        processed = process_func(*slices, **kwargs)  # 现在slices未必只有1个张量，所以是*slices。然后后面的**kwargs不变。
        if not isinstance(processed, (tuple, list)):  # 如果output_slice不是tuple或者list类别，就变成list。
            processed = [processed]
        processed_all.append(processed)
    processed_all_zipped = list(zip(*processed_all))
    # 上句，processed_all_zipped是个list，里面的每一个元素都是tuple（这个时候好像不太方便给他变成np.array，毕竟这个函数
    #     不知道process_func的输入输出，也就不知道他有几个元素啊）。
    result = [np.stack(o, axis=0) for o in zip(processed_all_zipped)]
    # 上句，弄成list，然后主函数里可以直接用，见调用的时候。但是调用的时候就会发现，比起单个输出的，就都多了一维。不过，可以在这儿就把它删掉。
    result = [np.squeeze(r) for r in result]  # 删掉多余的维度（是前面的zip等操作搞出来的）
    if len(result) == 1:
        result = result[0]
    return result

def find_unique_or_most_confident_detections(input_labels_and_score, class_num):
    """
    跟上面的函数有点像，只不过这儿是，如果某一个探测结果是某个非背景类的唯一的一个，
        或者虽然不唯一但是是mrcnn得分最大的（合理吗？是不是mrcnn非零类得分+rpnd得分最大的好一些？），就都把相应位置标记出来。
    输入的input_class_labels是n*2向量，n就是这张图中探测结果的总数，然后一列是类别，一列是得分。第0类是背景类，不考虑。
    """
    class_labels = input_labels_and_score[:,0]
    class_lprobs = input_labels_and_score[:,1]
    mrcnn_probs = input_labels_and_score[:,2]  # 加的。。
    unique = np.zeros_like(class_labels, dtype=bool)
    for i in range(1, class_num):  # 第0类是背景类，从第1类开始。
        ixi = np.where(class_labels==i)[0]  # 属于这一类的探测结果索引号组成的向量，如，[0, 1, 25]
        if ixi.shape[0] == 1:  # 是否只有1个探测结果，是这个标签的？
            unique[ixi] = True
        elif ixi.shape[0] >= 2:  # 有2个以上的探测结果
            most_confident_pos_in_ix = np.argmax(class_lprobs[ixi])  # 是在ixi中的位置
            unique[ixi[most_confident_pos_in_ix]] = True  # 如果不止1个，那么，得分最大的予以保留。
            # 加一个，如果mrcnn得分大于0.999且别的这一类的mrcnn得分都小于0.95，那么也保留。就是为了split2的13的。。
            highest_mrcnn_pos = np.argmax(mrcnn_probs[ixi])
            highest_mrcnn = np.max(mrcnn_probs[ixi])
            mrcnn_diff = highest_mrcnn-mrcnn_probs[ixi]  # 最大得分和其他得分的差值
            mrcnn_diff_sort = np.sort(mrcnn_diff)
            next_highest_diff = mrcnn_diff_sort[1]  # 第二大的得分，和最大得分的差值。
            if highest_mrcnn>0.999 and next_highest_diff > 0.1:
                unique[ixi[highest_mrcnn_pos]] = True
        else:  # 没有这一类的探测结果
            pass
    return unique

def select_by_rpn_and_mrcnn(input_detections, keep, line, threshold):
    """根据rpn和mrcnn的结果删掉一些不太可能的探测结果。
    MRCNN的探测结果，会有下面的问题：
        MRCNN程序做测试的时候，是直接先把背景类的都删了（有一句keep = tf.where(class_ids > 0)[:, 0]），
        然后选出信心分值大于DETECTION_MIN_CONFIDENCE阈值的（有一句conf_keep =
            tf.where(class_ scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]），
        然后选出来这两类的交集，也就是说是前景类、且信心分值大于DETECTION_MIN_CONFIDENCE阈值的。
        这样的话，有可能一些分类错误（如第1类给判断成第0类即背景）的就检测不到了，
        也有可能有好几个信心分值的大于那个阈值的，就多弄出来了。
    这个函数主要解决“多弄出来”的问题。具体地，删除rpn信心分值小于threshold的mrcnn探测结果，
        但是，要保留每一类的、mrcnn得分最大的那一个，因为有的时候，mrcnn预测某一类的rpn分值都不大于threshold，
            这种情况虽然少，但是还是要留意一下。最后的输入keep就相当于是这个免死金牌。
    """
    ix = np.where(input_detections[:, line] > threshold)[0]
    ix1 = np.where(keep==True)[0]
    ix_all = np.array(list(set(ix).union(set(ix1))))  # 以上两个np.array的“并集”。
    # 要先变成变成set，再变回list，再变回np.array。
    selected = input_detections[ix_all, :]
    # sorted = np.sort(selected, axis=0)  # 注意不能用这个，会把所有的列都从小到大排序的。
    sorted = selected[selected[:, 0].argsort()]  # 按照第一列排序
    sorted_ix = ix_all[selected[:, 0].argsort()]  # 按照第一列排序。注意这儿要多一个[0]，因为np.where输出的是tuple。
    P = input_detections.shape[0] - sorted.shape[0]
    padded = np.pad(sorted, ((0, P), (0, 0)),'constant',constant_values = (0))  # 必须写((0, P), (0, 0))，
    # 而不能是((0, P))，因为((0, P))相当于是((0, P), (0, P))！
    padded_ix = np.pad(sorted_ix, ((0, P)), 'constant', constant_values=(-1))  # 必须写((0, P))，
    # 而不能是((0, P), (0, 0))，因为这个时候是1维的。然后要补-1而不是0，因为ix是本来就可能取0的。
    return padded, padded_ix

def select_mrcnn_by_rpn(input, ix):
    """这个是处理一张图的函数，先去掉ix里补的-1，然后选出来，然后再补0。"""
    ix_unpadded = np.where(ix > -1)[0]  # ix中非补-1的位置
    pad_len = input.shape[0] - ix_unpadded.shape[0]
    output = input[ix[ix_unpadded]]
    # 上句，原来是input[ix[ix_unpadded], :]，但是考虑到input的维度可能不一样，就删掉了那个“,:”，这样，只让他在第0维度挑选就可以了。
    #     比如，下面处理_mrcnn_boxes_and_scores_all_test的时候，输入是(100, 6)，而处理aver_score的时候，输入是(100,)。
    if len(output.shape) == 2:
        padded_output = np.pad(output, ((0, pad_len), (0, 0)), 'constant', constant_values=(0))
    elif len(output.shape) == 1:
        padded_output = np.pad(output, ((0, pad_len)), 'constant', constant_values=(0))
    else:
        padded_output = output
        print('输入维度不是1维或者2维，没有补零，留意是否有问题。')
    return padded_output

def vote_by_score_and_coodinates(aver_scores_this_label, previous_non_BG_label, next_non_BG_label,
                                 labels_trimmed, boxes_trimmed, this_label, boxes_this_label):
    """通过外界输入的得分、坐标值来打分，判断一个探测结果是不是正例。
    输入：
    aver_scores_this_label：这个标签的所有探测结果的rpn/mrcnn平均值；
    previous_non_BG_label：前一个非零标签
    next_non_BG_label：后一个非零标签
    labels_trimmed：所有探测结果的标签
    boxes_trimmed：所有探测结果的外接矩形
    this_label：当前探测结果的标签
    boxes_this_label：当前探测结果的外接矩形
    后来发现，这个东西训练的时候肯定用不到；测试的时候，如果有了那个SG滤波删除容易的假正例，可能就也没啥用。。。
    """
    scores_selection = np.zeros_like(aver_scores_this_label, dtype=np.float32)  # 投票得分，初始化为全0的
    aver_scores_sorted = np.sort(aver_scores_this_label)  # 输入的得分，这儿是rpn/mrcnn信心分值。现在从小到大排列。
    aver_scores_sorted = aver_scores_sorted[::-1]  # 倒序过来，从大到小。
    aver_scores_sorted_pos = np.argsort(aver_scores_this_label)
    # 所有认为是这一类的东西，按照信心分值从小到大排列后的索引号（注意是在same_label里的索引号）顺序.
    aver_scores_sorted_pos = aver_scores_sorted_pos[::-1]  # 倒序过来，从大到小。
    for i2 in range(aver_scores_this_label.shape[0]):
        if i2 == 0:
            pass  # 信心分值最大的，保留为0分
        else:
            score_diff = aver_scores_sorted[0] - aver_scores_sorted[i2]
            # 这一个被认为是L4的，分值比最高的低了多少。
            scores_selection[aver_scores_sorted_pos[i2]] = scores_selection[aver_scores_sorted_pos[i2]] - score_diff
            # 扣除这个分值。
    if aver_scores_sorted[0] > aver_scores_sorted[1] + 0.2:
        # 如果信心分值最大的，比信心分值第二大的大了0.2以上，说明信心很充分的。
        scores_selection[aver_scores_sorted_pos[0]] = scores_selection[aver_scores_sorted_pos[0]] + 3  # 直接投3票
    elif aver_scores_sorted[0] > aver_scores_sorted[1] + 0.1:
        # 如果信心分值最大的，比信心分值第二大的大了0.1以上，说明信心也挺充分的。
        scores_selection[aver_scores_sorted_pos[0]] = scores_selection[aver_scores_sorted_pos[0]] + 2  # 直接投2票
    elif aver_scores_sorted[0] > aver_scores_sorted[1] + 0.05:
        # 如果信心分值最大的，比信心分值第二大的大了0.05以上，说明还有信心。
        scores_selection[aver_scores_sorted_pos[0]] = scores_selection[aver_scores_sorted_pos[0]] + 1  # 投1票
    elif aver_scores_sorted[0] > 0.99:
        low_score = np.where(aver_scores_sorted <= aver_scores_sorted[0] - 0.05)[0]
        scores_selection[low_score] = scores_selection[low_score] - 0.5
        # 分数有点低（差了0.05），那么投-0.5票
        too_low_score = np.where(aver_scores_sorted <= aver_scores_sorted[0] - 0.1)[0]
        scores_selection[too_low_score] = scores_selection[too_low_score] - 0.5
        # 分数特别低（差了0.1），那么再投-0.5票，一共-1票。
    else:
        pass
    if previous_non_BG_label is not None and previous_non_BG_label == this_label + 1:  # 这儿是+1，因为前面的标签大
        previous_label = np.where(labels_trimmed == previous_non_BG_label)[0]
        if len(previous_label) == 1:  # 如果前面的正例标签只有1个
            previous_box = boxes_trimmed[previous_label]
            y1p, x1p, y2p, x2p = previous_box[0]  # 上一个的四个坐标
            y1 = boxes_this_label[:, 0]  # 这一个的四个坐标，y1等都至少是2个数。
            x1 = boxes_this_label[:, 1]
            y2 = boxes_this_label[:, 2]
            x2 = boxes_this_label[:, 3]
            # 此处和滑脱那个不一样的是，这儿直接用x/y坐标位置、宽度、高度去做，就不用那个
            #     “y1_L4_nearest > np.max(y1_L4_others) and y2_L4_nearest > np.max(y2_L4_others)
            #      and y2_L4_nearest < y1_L5”这个玩意了。
            delta_x1 = abs(x1p - x1)  # 外接矩形左边的x坐标，和上一个正例外接矩形左边的x坐标差值
            delta_x2 = abs(x2p - x2)  # 外接矩形右边的x坐标，和上一个正例外接矩形右边的x坐标差值
            width_p = x2p - x1p  # 上一个正例的宽度
            width = x2 - x1  # 所有这个类别的正例的宽度
            delta_y1 = abs(y1p - y1)  # 外接矩形上边的y坐标，和上一个正例外接矩形上边的y坐标差值
            delta_y2 = abs(y2p - y2)  # 外接矩形下边的y坐标，和上一个正例外接矩形下边的y坐标差值
            height = y2p - y1p  # 上一个正例的高度
            height_p = y2 - y1  # 所有这个类别的正例的宽度
            somewhat_far_pos_x1 = np.where(delta_x1 > 0.8 * width_p)[0]  # “x1坐标相对于上一个椎骨偏离多”的那几个
            too_far_pos_x1 = np.where(delta_x1 > 2 * 0.8 * width_p)[0]  # 类似
            scores_selection[somewhat_far_pos_x1] = scores_selection[somewhat_far_pos_x1] - 0.5  # 投-0.5票
            scores_selection[too_far_pos_x1] = scores_selection[too_far_pos_x1] - 0.5  # 再投-0.5票
            somewhat_far_pos_x2 = np.where(delta_x2 > 0.8 * width_p)[0]  # “x2坐标相对于上一个椎骨偏离多”的那几个
            too_far_pos_x2 = np.where(delta_x2 > 2 * 0.8 * width_p)[0]  # 类似
            scores_selection[somewhat_far_pos_x2] = scores_selection[somewhat_far_pos_x2] - 0.5  # 投-0.5票
            scores_selection[too_far_pos_x2] = scores_selection[too_far_pos_x2] - 0.5  # 再投-0.5票
            """上面8句，考察这一类的椎骨角点坐标的横向偏离：按理说，x方向应该和前一个椎骨的x坐标差不多，所以，
                用椎骨的宽度的0.8倍作为横向偏离的阈值（似乎即使滑脱的，也没差这么多），
            因此，超过0.8就认为有点不对了，超过1.6就认为很不对了。
                就用投票法，给超过这些阈值的减掉票数，以促进保留更合适的。
            """
            too_wide = np.where(width > 2 * width_p)[0]  # 宽度大于上面椎骨宽度的2倍
            too_thin = np.where(width < 0.5 * width_p)[0]  # 宽度小于上面椎骨宽度的0.5倍
            scores_selection[too_wide] = scores_selection[too_wide] - 1  # 直接投-1票
            scores_selection[too_thin] = scores_selection[too_thin] - 1  # 直接投-1票
            """上面4句，考察这一类的椎骨的宽度。如果比起上一个椎骨宽度大了2倍或者小于0.5倍，
            基本可以判定有问题。
            """
            somewhat_far_pos_y1 = np.where(delta_y1 > 1.5 * height_p)[0]
            # y1相对于上一个椎骨偏离，和x1的不一样，见下。
            scores_selection[somewhat_far_pos_y1] = scores_selection[somewhat_far_pos_y1] - 0.5  # 投-0.5票
            somewhat_far_pos_y2 = np.where(delta_y2 > 1.5 * height_p)[0]
            # y2相对于上一个椎骨偏离，和x1的不一样，见下。
            scores_selection[somewhat_far_pos_y2] = scores_selection[somewhat_far_pos_y2] - 0.5  # 投-0.5票
            smallest_y1_diff = np.min(delta_y1)  # y1坐标相对于下一个椎骨最近的那个
            smallest_y2_diff = np.min(delta_y2)  # y2坐标相对于下一个椎骨最近的那个
            somewhat_far_pos_y1_1 = np.where(delta_y1 > 0.9 * height_p + smallest_y1_diff)[0]
            scores_selection[somewhat_far_pos_y1_1] = scores_selection[somewhat_far_pos_y1_1] - 0.5  # 投-0.5票
            somewhat_far_pos_y2_1 = np.where(delta_y2 > 0.9 * height_p + smallest_y2_diff)[0]
            scores_selection[somewhat_far_pos_y2_1] = scores_selection[somewhat_far_pos_y2_1] - 0.5  # 投-0.5票
            """上面4句，考察这一类的椎骨角点坐标的纵向偏离：按理说，y方向应该和前一个椎骨的y坐标
                正好相差一个椎骨高度，大约0.1左右、即50个像素点（和x方向的不一样）。所以，如果超过了1.5个椎骨高度，就认为可能有问题。
                然后，其实如果差得太少了（如小于0.5个椎骨高度）也不行，但是，这种情况可以用重合度搞定，这儿就不弄了。
            类似于x坐标，也用投票法，给超过这些阈值的减掉票数，以促进保留更合适的。
            """
            too_tall = np.where(height > 2 * height_p)[0]  # 高度大于上面椎骨高度的2倍
            too_short = np.where(height < 0.5 * height_p)[0]  # 高度小于上面椎骨高度的0.5倍
            scores_selection[too_wide] = scores_selection[too_wide] - 1  # 直接投-1票
            scores_selection[too_thin] = scores_selection[too_thin] - 1  # 直接投-1票
            """上面4句，考察这一类的椎骨的高度，和刚才考察宽度类似。"""
    if next_non_BG_label is not None and next_non_BG_label == this_label - 1:
        next_label = np.where(labels_trimmed == next_non_BG_label)[0]
        if len(next_label) == 1:  # 如果后面的正例标签只有1个
            next_box = boxes_trimmed[next_label]
            y1n, x1n, y2n, x2n = next_box[0]  # 下一个的四个坐标
            y1 = boxes_this_label[:, 0]  # 这一个的四个坐标，y1等都至少是2个数。
            x1 = boxes_this_label[:, 1]
            y2 = boxes_this_label[:, 2]
            x2 = boxes_this_label[:, 3]
            delta_x1 = abs(x1n - x1)  # 外接矩形左边的x坐标，和下一个正例外接矩形左边的x坐标差值
            delta_x2 = abs(x2n - x2)  # 外接矩形右边的x坐标，和下一个正例外接矩形右边的x坐标差值
            width_n = x2n - x1n  # 下一个正例的宽度
            width = x2 - x1  # 所有这个类别的正例的宽度
            delta_y1 = abs(y1n - y1)  # 外接矩形上边的y坐标，和下一个正例外接矩形上边的y坐标差值
            delta_y2 = abs(y2n - y2)  # 外接矩形下边的y坐标，和下一个正例外接矩形下边的y坐标差值
            height_n = y2n - y1n  # 上一个正例的高度
            height = y2 - y1  # 所有这个类别的正例的宽度
            somewhat_far_pos_x1 = np.where(delta_x1 > 0.8 * width_n)[0]  # “x1坐标相对于下一个椎骨偏离多”的那几个
            too_far_pos_x1 = np.where(delta_x1 > 2 * 0.8 * width_n)[0]  # 类似
            scores_selection[somewhat_far_pos_x1] = scores_selection[somewhat_far_pos_x1] - 0.5  # 投-0.5票
            scores_selection[too_far_pos_x1] = scores_selection[too_far_pos_x1] - 0.5  # 再投-0.5票
            somewhat_far_pos_x2 = np.where(delta_x2 > 0.8 * width_n)[0]  # “x2坐标相对于下一个椎骨偏离多”的那几个
            too_far_pos_x2 = np.where(delta_x2 > 2 * 0.8 * width_n)[0]  # 类似
            scores_selection[somewhat_far_pos_x2] = scores_selection[somewhat_far_pos_x2] - 0.5  # 投-0.5票
            scores_selection[too_far_pos_x2] = scores_selection[too_far_pos_x2] - 0.5  # 再投-0.5票
            """上面8句，考察这一类的椎骨角点坐标的横向偏离：按理说，x方向应该和后一个椎骨的x坐标差不多，所以，
                用椎骨的宽度的0.8倍作为横向偏离的阈值（似乎即使滑脱的，也没差这么多），
            因此，超过0.8就认为有点不对了，超过1.6就认为很不对了。
                就用投票法，给超过这些阈值的减掉票数，以促进保留更合适的。
            """
            too_wide = np.where(width > 2 * width_n)[0]  # 宽度大于下一个椎骨宽度的2倍
            too_thin = np.where(width < 0.5 * width_n)[0]  # 宽度小于下一个椎骨宽度的0.5倍
            scores_selection[too_wide] = scores_selection[too_wide] - 1  # 直接投-1票
            scores_selection[too_thin] = scores_selection[too_thin] - 1  # 直接投-1票
            """上面4句，考察这一类的椎骨的宽度。如果比起下一个椎骨宽度大了2倍或者小于0.5倍，
            基本可以判定有问题。
            """
            somewhat_far_pos_y1 = np.where(delta_y1 > 1.5 * height_n)[0]  # y1相对于下一个椎骨偏离，和x1的不一样，见下。
            scores_selection[somewhat_far_pos_y1] = scores_selection[somewhat_far_pos_y1] - 0.5  # 投-0.5票
            somewhat_far_pos_y2 = np.where(delta_y2 > 1.5 * height_n)[0]  # y2相对于下一个椎骨偏离，和x1的不一样，见下。
            scores_selection[somewhat_far_pos_y2] = scores_selection[somewhat_far_pos_y2] - 0.5  # 投-0.5票
            smallest_y1_diff = np.min(delta_y1)  # y1坐标相对于下一个椎骨最近的那个
            smallest_y2_diff = np.min(delta_y2)  # y2坐标相对于下一个椎骨最近的那个
            somewhat_far_pos_y1_1 = np.where(delta_y1 > 0.9 * height_n + smallest_y1_diff)[0]
            scores_selection[somewhat_far_pos_y1_1] = scores_selection[somewhat_far_pos_y1_1] - 0.5  # 投-0.5票
            somewhat_far_pos_y2_1 = np.where(delta_y2 > 0.9 * height_n + smallest_y2_diff)[0]
            scores_selection[somewhat_far_pos_y2_1] = scores_selection[somewhat_far_pos_y2_1] - 0.5  # 投-0.5票
            # smallest_y1_diff_pos = np.argmin(delta_y1)  # 最近的位置
            # smallest_y2_diff_pos = np.argmin(delta_y2)
            # if smallest_y1_diff_pos == smallest_y2_diff_pos and \
            #     np.max(delta_y1-smallest_y1_diff) > 0.9 * height_n and \
            #     np.max(delta_y2-smallest_y2_diff) > 0.9 * height_n:
            #     scores_selection[smallest_y1_diff_pos] = scores_selection[smallest_y1_diff_pos] + 1
            """上面4句，考察这一类的椎骨角点坐标的纵向偏离：按理说，y方向应该和下一个椎骨的y坐标
                正好相差一个椎骨高度（和x方向的不一样）。所以，如果超过了1.5个椎骨高度，就认为可能有问题。
                然后，其实如果差得太少了（如小于0.5个椎骨高度）也不行，但是，这种情况可以用重合度搞定，这儿就不弄了。
            类似于x坐标，也用投票法，给超过这些阈值的减掉票数，以促进保留更合适的。
            然后新加的是要判断会不会把L3当成L4的那种情况（因为有的时候还真不会差1.5 * height_n，
                其实有时候也就1个height_n左右）。所以，想加一下，比如说有3个都是第1类的，
                然后肯定有一个是离第2类最接近的吧，就看其他几个比最接近的这个（smallest_y1_diff/smallest_y2_diff）
                远了多少，如果远了0.9 * height_n以上，就认为是有点远了，就扣0.5分。
            """
            too_tall = np.where(height > 2 * height_n)[0]  # 高度大于下一个椎骨高度的2倍
            too_short = np.where(height < 0.5 * height_n)[0]  # 高度小于下一个椎骨高度的0.5倍
            scores_selection[too_wide] = scores_selection[too_wide] - 1  # 直接投-1票
            scores_selection[too_thin] = scores_selection[too_thin] - 1  # 直接投-1票
            """上面4句，考察这一类的椎骨的高度，和刚才考察宽度类似。"""
    return scores_selection

def select_detections_for_mp_1st_round(num_classes, RPN_MIN_CONFIDENCE_MP, _rpn_scores_nms, _rpn_rois, _mrcnn_boxes_and_scores, _mrcnn_class):
    """把探测框架输出的东西排成一个序列放到crf里去，第1轮选取。
    和下面主函数里的一部分很像，不过没仔细看是否完全一样的。不管他了。"""
    _rpn_scores_nms1 = np.expand_dims(_rpn_scores_nms, axis=2)  # 准备存起来rpn结果
    _rpn_rois_and_scores = np.concatenate([_rpn_rois, _rpn_scores_nms1], axis=2)
    _mrcnn_boxes_and_scores_full = np.concatenate([_mrcnn_boxes_and_scores, _mrcnn_class], axis=2)
    _mrcnn_labels = _mrcnn_boxes_and_scores_full[:, :, 4]
    _mrcnn_probs = _mrcnn_boxes_and_scores_full[:, :, 5]
    _mrcnn_probs = _mrcnn_probs * _mrcnn_labels.astype(bool)  # 如果mrcnn判断是背景类，那么得分就乘以0，就改成0。
    _rpn_score = _rpn_rois_and_scores[:, :, 4]
    _aver_score = (_mrcnn_probs + _rpn_score) / 2  # mrcnn和rpn平均得分。
    labels_and_probs = np.stack([_mrcnn_labels, _aver_score, _mrcnn_probs], axis=2)  # mrcnn的标签和(mrcnn+rpn)平均得分和mrcnn得分
    keep_unique_or_most_confident = batch_processing \
        (process_func=find_unique_or_most_confident_detections,
         input_batch=labels_and_probs,
         class_num=num_classes)
    # 上句，对于每张图中的每种非背景类标签搜索，如果某一类只有1个探测结果，或者虽然不是1个探测结果但是这个结果是rpn/mrcnn平均得分最高的，
    #     那么，就在这个探测结果的位置给放一个标签，这样，就不会在下一句筛选的时候被筛掉。
    rpn_positive_threshold = RPN_MIN_CONFIDENCE_MP  # 也用这个0.9吧。
    line_representing_rpn_score = 4
    _rpn_rois_and_scores_selected_test_1st_round, ix_selected_test_by_rpn_rois = \
        batch_processing_multi_input_and_output \
            (process_func=select_by_rpn_and_mrcnn,
             input_batches=[_rpn_rois_and_scores, keep_unique_or_most_confident],
             line=line_representing_rpn_score,
             threshold=rpn_positive_threshold)
    aver_score_1st_round = batch_processing_multi_input_and_output \
        (process_func=select_mrcnn_by_rpn,
         input_batches=[_aver_score, ix_selected_test_by_rpn_rois])
    # 上句，第1轮选取后，选出来的rpn/mrcnn平均分。
    _mrcnn_boxes_and_full_scores_1st_round = batch_processing_multi_input_and_output \
        (process_func=select_mrcnn_by_rpn,
         input_batches=[_mrcnn_boxes_and_scores_full, ix_selected_test_by_rpn_rois])
    # 上句，第1轮选取后，选出来的mrcnn外接矩形、类别、分数、其他类别的分数。
    ix_selected_test_by_rpn_rois = ix_selected_test_by_rpn_rois.astype(np.int32)
    return _mrcnn_boxes_and_full_scores_1st_round, aver_score_1st_round, ix_selected_test_by_rpn_rois

def select_detections_for_mp_1st_round_tf(config, rpn_scores_nms, rpn_rois, mrcnn_boxes_and_scores, mrcnn_class):
    """就是把上面那个给包裹成tf的。"""
    mrcnn_boxes_and_full_scores_1st_round, aver_score_1st_round_tf, ix_selected_test_by_rpn_rois_tf = \
        tf.py_func(select_detections_for_mp_1st_round,
                   [config.NUM_CLASSES, config.RPN_MIN_CONFIDENCE_MP, rpn_scores_nms, rpn_rois, mrcnn_boxes_and_scores, mrcnn_class],
                   [tf.float32, tf.float32, tf.int32])
    return mrcnn_boxes_and_full_scores_1st_round, aver_score_1st_round_tf, ix_selected_test_by_rpn_rois_tf

def select_detections_for_mp_2nd_round(_mrcnn_boxes_and_full_scores_1st_round_this, aver_score_1st_round_this,
                                        IOU_threshold_one, IOU_threshold_two, class_id_min, class_id_max):
    """把探测框架输出的东西排成一个序列放到crf里去，第2轮选取。
    和下面主函数里的一部分很像，不过没仔细看是否完全一样的。不管他了。
    和上面的函数不一样的是，这儿输入的是一张图，而上面的第1轮选取输入的是一个批次。
    后来发现，这个东西训练的时候肯定用不到；测试的时候，如果有了那个SG滤波删除容易的假正例，可能就也没啥用。。。
    """
    boxes_in_this_image = _mrcnn_boxes_and_full_scores_1st_round_this[:, :4]
    labels_in_this_image = _mrcnn_boxes_and_full_scores_1st_round_this[:, 4]
    aver_scores_in_this_image = aver_score_1st_round_this
    boxes_trimmed, ix_unpadded = trim_zero(boxes_in_this_image)
    labels_trimmed = labels_in_this_image[ix_unpadded]
    aver_scores_trimmed = aver_scores_in_this_image[ix_unpadded]
    non_BG = np.where(labels_trimmed > 0)[0]  # 前景类的位置
    IOUs = MRCNN_utils.compute_overlaps(boxes_trimmed, boxes_trimmed)
    box_num = boxes_trimmed.shape[0]
    for j in range(box_num):
        IOUs[j, j] = 0  # 对角线元素原来都是1（自己和自己的IoU，现在都清零）
    keep_iou = np.ones_like(labels_trimmed, dtype=bool)  # 一开始初始化为全保留，然后根据IoU删掉。
    tested = np.zeros_like(labels_trimmed, dtype=bool)  # 如果已经检测过了，就不必再检测了。
    for i1 in range(box_num):
        if tested[i1] == True:
            pass
        else:
            if labels_trimmed[i1] == 0:  # 如果是背景类
                max_IOU = np.max(IOUs[i1, non_BG])  # 和非背景类的重合度最大值
                tested[i1] = True
                previous_non_BG_label = None
                next_non_BG_label = None
                for i1_1 in range(i1, -1, -1):  # range(i1,-1,-1)应该是从i1开始一一个往下减，减到0为止（-1不算）
                    if labels_trimmed[i1_1] != 0:  # 如果前面有非背景类，那么记录下来这个标签（并且停止搜索），否则保留None。
                        previous_non_BG_label = labels_trimmed[i1_1]
                        break
                for i1_2 in range(i1, box_num):
                    if labels_trimmed[i1_2] != 0:  # 如果后面有非背景类，那么记录下来这个标签（并且停止搜索），否则保留None。
                        next_non_BG_label = labels_trimmed[i1_2]
                        break
                """以上，得到这个样例的前一个/后一个非背景类的类别，所谓前一个/后一个就是按照y坐标排好了的。"""
                if max_IOU > IOU_threshold_one:
                    # 上面的if：如果mrcnn认为某个预测外接矩形是背景类，且和任意一个非背景类的外接矩形重合度过高（大于0.25），
                    #    就认为mrcnn判断正确，确实是背景类（相当于它和某个前景类外接矩形是同一块椎骨，但是搞偏了），把它删除。
                    keep_iou[i1] = False
                elif len(np.where(IOUs[i1, non_BG] > IOU_threshold_two)[0]) >= 2:
                    # 上面的if：如果mrcnn认为某个预测外接矩形是背景类，且和某两个非背景类的外接矩形重合度都有点高（都大于0.15），
                    #    就认为mrcnn判断正确，确实是背景类（相当于它搭在了某两个非背景类外接矩形之间），把它删除。
                    keep_iou[i1] = False
                elif (previous_non_BG_label is not None) and (next_non_BG_label is not None) \
                        and next_non_BG_label == previous_non_BG_label - 1:
                    # 上面的if，如果mrcnn认为某个预测外接矩形是背景类，且后面的前景类类别序号正好比前面的序号小1，
                    #     那就说明前后两个前景类都已经有比它更可靠的探测结果了，那么，这个就很可能真的是背景类。
                    # （现在还没想好，到底是直接就认为是背景类了，还是通过降低那个IOU_threshold_one的方法，看看再说。。。）
                    keep_iou[i1] = False
                elif (class_id_max in labels_trimmed) or (class_id_min in labels_trimmed):
                    # 上面的if：如果在这张图里发现了最大标签（比如说，如果标签最大就是10，而这张图里已经发现了10），
                    # 那么，如果mrcnn认为某个预测外接矩形是背景类，且这个预测外接矩形的y坐标在那个最大标签的预测结果下方，
                    # 就认为它是真的背景类，把它删除。
                    # 或者，如果在这张图里发现了最小标签（比如说，如果标签最大就是1，而这张图里已经发现了1），
                    # 那么，如果mrcnn认为某个预测外接矩形是背景类，且这个预测外接矩形的y坐标在那个最小标签的预测结果上方，
                    # 就认为它是真的背景类，把它删除。
                    if (class_id_max in labels_trimmed):
                        last = np.where(labels_trimmed == class_id_max)[0][0]
                        # 上句，标签为最大标签的那个检测结果（肯定是mrcnn正例）
                        last_y1 = boxes_trimmed[last, 0]
                        # 上句，标签为最大标签的那个检测结果的上边缘y值
                        this_y1 = boxes_trimmed[i1, 0]
                        # 上句，当前检测结果（mrcnn背景类）的上边缘y值
                        if this_y1 < last_y1:
                            keep_iou[i1] = False
                            # 如果这个检测结果（mrcnn认为是背景类）在最大标签的那个检测结果的上方，
                            # 就认为它真的是背景类。
                    if (class_id_min in labels_trimmed):
                        first = np.where(labels_trimmed == class_id_min)[0][0]
                        first_y1 = boxes_trimmed[first, 0]
                        this_y1 = boxes_trimmed[i1, 0]
                        if this_y1 > first_y1:
                            keep_iou[i1] = False
                            # 如果这个检测结果（mrcnn认为是背景类）在最小标签的那个检测结果的下方，
                            # 就认为它真的是背景类（注意现在最小标签的检测结果是在最下面的）。
            else:  # 如果是前景类
                same_label = np.where(labels_trimmed == labels_trimmed[i1])[0]  # 同一个前景类的
                this_label = labels_trimmed[i1]
                boxes_this_label = boxes_trimmed[same_label]  # 这个前景类的探测结果外接矩形
                aver_scores_this_label = aver_scores_trimmed[same_label]  # 这个探测结果的平均得分（注意，不是mrcnn得分），
                # 因为我发现有个现象，如果把L3误认为L4，那么，有可能L3的mrcnn得分比真的L4还大，但是它的rpn得分就有点低
                # （比如说L4的rpn得分可能是0.999，而它只有0.95什么的）
                # 这个是选取这些正例用的得分，类似于滑脱的那个程序
                if len(same_label) == 1:  # 如果这一类只有1个mrcnn判断为正例
                    tested[i1] = True
                    pass
                else:
                    previous_non_BG_label = None
                    next_non_BG_label = None
                    for i1_1 in range(i1, -1, -1):  # range(i1,-1,-1)应该是从i1开始一一个往下减，减到0为止（-1不算）
                        if labels_trimmed[i1_1] != 0 and labels_trimmed[i1_1] != labels_trimmed[i1]:
                            # 如果前面有非背景类、且标签不等于这个标签，那么记录下来这个标签（并且停止搜索），否则保留None。
                            previous_non_BG_label = labels_trimmed[i1_1]
                            break
                    for i1_2 in range(i1, box_num):
                        if labels_trimmed[i1_2] != 0 and labels_trimmed[i1_2] != labels_trimmed[i1]:
                            # 如果后面有非背景类、且标签不等于这个标签，那么记录下来这个标签（并且停止搜索），否则保留None。
                            next_non_BG_label = labels_trimmed[i1_2]
                            break
                    """以上，得到这个样例的前一个/后一个非背景类的类别，所谓前一个/后一个就是按照y坐标排好了的。"""
                    for s1 in range(same_label.shape[0]):  # 在具有相同标签的外接矩形中循环
                        if tested[same_label[s1]] == False:
                            tested[same_label[s1]] = True
                            for s2 in range(s1 + 1, same_label.shape[0]):  # 遍历当前外接矩形之后的、所有具有相同标签的外接矩形
                                IoU_this = IOUs[same_label[s1], same_label[s2]]
                                if IoU_this > IOU_threshold_one:  # 如果当前外接矩形和它之后的某个同标签的外接矩形的IoU大于这个阈值
                                    scores_selection = vote_by_score_and_coodinates\
                                        (aver_scores_this_label, previous_non_BG_label, next_non_BG_label,
                                         labels_trimmed, boxes_trimmed, this_label, boxes_this_label)
                                    """以上，类似于滑脱的方法，给同一个标签的评分。如果是IoU比较大的，就用上面的得分去掉一些；如果IoU小的，则不管。"""
                                    if scores_selection[s1] > scores_selection[s2]:
                                        # 只保留信心分值大的，且把被淘汰的标记为“已检测”（当前的外接矩形已经被标记为“已检测”了）。
                                        # 未被淘汰的不标记为“已检测”，因为我们并不知道以后是否会有别的外接矩形把它淘汰。
                                        keep_iou[same_label[s2]] = False
                                        tested[same_label[s2]] = True
                                    else:
                                        keep_iou[same_label[s1]] = False
                                        tested[same_label[s1]] = True
                                else:  # 如果当前外接矩形和其他同标签的外接矩形IoU不大，那么不管他（认为可能确实是正例，但是给分类错了）
                                    pass
                                """注意，这儿和滑脱的那个不一样，是IoU比较大的、且同一个标签的，才投票、算分数，选出来分数大的而删掉分数小的；
                                如果IoU比较小，就都保留着，因为有可能是这两个确实都是正例，只不过标签错了。比如说，
                                    好像测试集里第14张图就是这样的，两个7标签，两个5标签，但是实际上应该是7 6 5 4这四个标签。
                                回忆一下思路是这样的：一开始做滑脱的时候，都没有这个投票分值的（见A46_crf_processing_3），
                                    就是发现IoU比较大，就用rpn/mrcnn平均分、即aver_scores_trimmed去删除一些，
                                    然后IoU不大的就没管他了。
                                后来，好像是有一次发现L3和L4都被认为是L4了。他俩重合度不大，就没法用IoU和aver_scores_trimmed区分开，
                                    所以才引入了那个投票，用坐标什么的去判断，然后把那个L3给弄成是背景类（也就是对应的keep_iou置零）；
                                但，现在我不需要把L3弄成背景类了，因为它有了属于它自己的一类，所以现在，如果再把L3认为是L4，
                                    那么也保留这个L3的探测结果，用CRF修正它的标签就好了。
                                """
                        else:
                            pass
    keep_iou1 = np.where(keep_iou == True)[0]
    pad_len = labels_in_this_image.shape[0] - keep_iou1.shape[0]
    keep_iou1_padded = np.pad(keep_iou1, ((0, pad_len)), 'constant', constant_values=(-1))
    _mrcnn_boxes_and_full_scores_2nd_round_this = _mrcnn_boxes_and_full_scores_1st_round_this[keep_iou1_padded, :]
    keep_iou1 = keep_iou1.astype(np.int32)
    return _mrcnn_boxes_and_full_scores_2nd_round_this, keep_iou1

def select_detections_for_mp_2nd_round_tf(mrcnn_boxes_and_full_scores_1st_round_this, aver_score_1st_round_this_tf,
                                        IOU_threshold_one, IOU_threshold_two, class_id_min, class_id_max):
    """就是把上面那个给包裹成tf的。"""
    mrcnn_boxes_and_full_scores_2nd_round_this, keep_iou1_tf = \
        tf.py_func(select_detections_for_mp_2nd_round,
                   [mrcnn_boxes_and_full_scores_1st_round_this, aver_score_1st_round_this_tf,
                    IOU_threshold_one, IOU_threshold_two, class_id_min, class_id_max],
                   [tf.float32, tf.int32])
    return mrcnn_boxes_and_full_scores_2nd_round_this, keep_iou1_tf