#coding=utf-8
# Copyright 2017 Zhongyi Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.6
import os
import time
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from nets import preprocessing, SpinePathNet,losses  # 预处理、网络、损失函数。
import tf_utils

# Fold is one of five folds cross-validation. 五折交叉验证
Fold = 5
# =========================================================================== #
# Model saving Flags. 模型存储的参数
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodels_gan_lstm_%s/'%Fold,  # 路径里可以加入变量的
    'Directory where checkpoints and event logs are written to.')  # 写入“程序执行节点”的路径
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1, 'GPU memory fraction to use.')

tf.app.flags.DEFINE_float(
    'weight', 1, 'The weight is between cross entropy loss and metric loss.')
# =========================================================================== #
# Dataset Flags. 数据集路径等参数
# =========================================================================== #

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 7, 'Number of classes to use in the dataset.')  # 7个类别
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train_%s_fold'%Fold, 'The name of the train/test split.')
    # 训练的时候，用'train_%s_fold'%Fold。spine_segmentation_train_7_class.tfrecord
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation/', 'The directory where the dataset files are stored.')  # 保存的那些.tfrecord文件的路径
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 202, 'The number of samples in the training set.')  # 训练集中的样本数
tf.app.flags.DEFINE_integer(
    'num_readers', 20,
    'The number of parallel readers that read data from the dataset.')  # 同时进行的阅读器数目（什么意思？李天洋也没管。。）
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')  # 用来构造批次的线程数（后面似乎没用到？）

# =========================================================================== #
# Optimization Flags. 优化算法的选择，以及相应的参数。
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',  # 最优化算法，rmsprop是随机梯度SDG的升级版，增加了一个衰减系数来控制历史信息的获取
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
"""
    关于各种优化算法:
    1、基础算法"sgd"：随机梯度下降，根据随机取得的样本，确定损失函数的梯度（沿着样本不同维度方向的下降速度），
        根据学习率和梯度去更新网络参数（各卷积层全连接层的权重偏执啥的），即“新参数=原参数-学习率*梯度”。
    2、动量算法"momentum"：加入了历史下降方向（相当于动量），避免网络参数更新“在山谷之间跳动”，解决病态梯度问题。
        更新变成了“新参数=原参数-学习率*梯度+动量参数*速度”。速度也是有更新的，原书上其实不太准，我觉得就是乘以了那个动量参数。
    3、自适应梯度"adagrad"：没有用那个动量项，但是给学习率除了个系数。具体地，全局学习率逐参数地除以历史梯度平方和的平方根，使得每个参数的学习率不同。
        更新变成了“新参数=原参数-学习率/累计平方梯度（加一个小常数避免分母太小）*梯度”。累计平方梯度也有更新：每次增加梯度平方和。
    4、二阶矩"rmsprop"：adagrad的升级版，更改了累计平方梯度的算法，让累计平方梯度更喜欢最近的梯度而渐渐丢弃遥远过去的梯度。
        更新仍然是“新参数=原参数-学习率/累计平方梯度（加一个小常数避免分母太小）*梯度”。但累计平方梯度更新不同：梯度平方和和当前累计平方梯度的加权和。
    5、"adam"：动量算法"momentum"和二阶矩"rmsprop"的结合，动量相当于是一阶矩。
        更新变成了“新参数=原参数-学习率/二阶矩（"rmsprop"的累计平方梯度加一个小常数）*一阶矩（动量项，和梯度有关，当前梯度和动量的加权平均）”。
    6、"adadelta"：自适应梯度"adagrad"的升级版，避免"adagrad"到最后梯度太小的问题。
        更新就是把3中的“学习率/累计平方梯度（加一个小常数避免分母太小）”的分子换成“当前参数的平方梯度”。
    7、"ftrl"：按照下面网址的理解，就是在"sgd"里又减了一个L1和L2正则项。
        http://vividfree.github.io/机器学习/2015/12/05/understanding-FTRL-algorithm
"""
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.94,  # adadelta（rmsprop也有这个参数）历史梯度平方的加权系数（也称为衰减系数，一般在0.9左右），而(1-这个数)是当前梯度平方和的加权系数。
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,  # adagrad算法中、初始的“累计梯度总量”。见《深度学习》书上算法8.4中的那个r项（书上写的是0）。
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')  # adam算法中，更新有偏一阶矩估计的那个ρ1。见《深度学习》书上算法8.7。
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')  # adam算法中，更新有偏二阶矩估计的那个ρ2。见《深度学习》书上算法8.7。
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')  # 似乎是各算法中中分母上加的那个小常数（问题为啥这么大？）。
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')  # FTRL算法的参数，可能和那个网址中2式的第一项有关。
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,  # FTRL算法的参数，可能是那个网址中2式的σs。
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')  # FTRL算法的L1正则项系数。
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')  # FTRL算法的L2正则项系数。
tf.app.flags.DEFINE_float(
    'momentum', 0.9,  # 动量优化算法"momentum"的动量项，《深度学习》书上算法8.2中的α。那个RMSPropOptimizer暂时不管他。
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')  # rmsprop里的动量项。应该是结合Nesterov动量的RMSProp，《深度学习》书上算法8.6中的α。
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')  # rmsprop里的衰减项，《深度学习》书上算法8.5中的ρ。

# =========================================================================== #
# Learning Rate Flags. 学习率的参数。
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',  # 学习率衰减形式：不变、指数、多项式。
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')  # 初始学习率
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,  # 最终学习率（不能一直衰减啊）
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')  # 标签平滑。不知道什么意思？？？
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')  # 学习率衰减因子
tf.app.flags.DEFINE_float( 
    'num_epochs_per_decay', 10.0,  # 多少个训练时代，做一次衰减
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,  # 滑动平均衰减
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,  # 最大训练步数
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_epochs', 500,  # 训练的时代数
                            'The number of training epochs.')

# =========================================================================== #
# Fine-Tuning Flags. 细调的路径
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'tmp/tfmodels_gan_lstm_%s/'%Fold,  # 好像是个预训练好的节点（那个checkpoint，网络参数）的路径。
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,  # 好像是和name_scope差不多，就是在变量名前面加上前缀。
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,  # 不要的预训练节点，不管了。
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,  # 不管了。
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', None,#['SpinePathNet/feature_embedding','SpinePathNet/logits/'],  # 恢复节点的时候，忽略没有的变量
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    
    if not FLAGS.dataset_dir:  # 检查是否有了数据集路径
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    ##logging tools
    tf.logging.set_verbosity(tf.logging.DEBUG)  # 输出日志信息的级别。
    # 一共五个级别，调试DEBUG，信息INFO，警告WARN，错误ERROR和致命FATAL。现在tf.logging.DEBUG是说只要是大于DEBUG的都输出。
    
    with tf.Graph().as_default():
        # 上局那个default似乎是默认的计算图。有的时候，程序里会有多个计算图。不过这儿似乎只有一个。
        
        # Create global_step.
        with tf.device('/cpu:0'):
            global_step = tf.train.create_global_step()
            """ 
            global_step（迄今为止训练了的批次数）往往会影响learning rate的计算，
                fine-tuning时，global_step肯定要从0开始记；继续训练时要从上次的断点开始计。
                这个网站https://www.zhihu.com/question/269968195说，读入一个节点（类似于model.ckpt-******的文件）之后，
                就用tf.train.get_or_create_global_step()这种命令得到已经执行了的训练批次数，作为此时的global_step值。
            后面做的时候，“if ckpt and ckpt.model_checkpoint_path:”那个地方
                从那个ckpt文件中（如果存在这个文件的话）读取了训练全局批次数；如果没有这个文件，就从头开始训练。
            """
        

        # Select the dataset. 数据集（.tfrecord文件）的名称。
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name) 
        
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                """
                上面name_scope 返回一个string，比如input。name_scope对“get_variable()创建的变量”的名字不会有任何影响，而创建的op会被加上前缀。
                    或者说，name_scope 是给op_name加前缀, variable_scope是给get_variable()创建的变量的名字加前缀。
                    比如说下面那个image的name就会是input/shuffle_batch:0一类的。那个shuffle_batch:0和这个input没关系，是别的地方给他取的。
                这个网站：https://blog.csdn.net/u012436149/article/details/53081454说了name_scope和variable_scope的事情。
                """
                filename_queue = tf.train.string_input_producer([dataset], num_epochs=FLAGS.num_epochs)
                """
                上面tf.train.string_input_producer是用来代替那个“使用placeholder占位，然后再feed_dict”的方法的。
                    见https://blog.csdn.net/zzk1995/article/details/54292859。
                1、使用tf.train.string_input_producer函数把我们需要的全部文件打包为queue类型（所谓filename_queue），之后tf开文件就从这个queue中取目录了，
                    要注意一点的是这个函数的shuffle参数默认是True，也就是你传给他文件顺序是1234，但是到时候读就不一定了，
                    我一开始每次跑训练第一次迭代的样本都不一样，还纳闷了好久，就是这个原因。
                2、那个网站后面还写了用reader读入数据、切开图像/标签对的方法。应该是在下句tf_utils.read_and_decode里。
                """
                image,mask_class,_ = tf_utils.read_and_decode(filename_queue, batch_size = FLAGS.batch_size,\
                                                                       capacity=20 * FLAGS.batch_size,\
                                                                       num_threads=FLAGS.num_readers,\
                                                                       min_after_dequeue=10 * FLAGS.batch_size, is_training=True)
                """
                上句就是把图像和标签给弄出来。正好和上面网站中说的“image和label一定要一起run出来”相呼应的。
                看了那个read_and_decode函数，返回值是图像本身、标签类别、超像素（舍弃）。
                执行一下看了，这个image的shape是(4, 512, 512, 1)，mask_class的shape是(4, 512, 512)
                    前者意义自明，后者应该就是这个批量中每一张图的、每一个像素点的标签。可以想象这个mask_class里的数应该是从0~6这7个数的。
                """
                labels = tf.to_float(tf.contrib.layers.one_hot_encoding(mask_class,FLAGS.num_classes))  # 标签变成1-热标签。此时shape是(4, 512, 512, 7)。
                mask_class_onehot_for_d = tf.to_float(tf.contrib.layers.one_hot_encoding(mask_class,FLAGS.num_classes, on_value=0.99,off_value=0.01))   # 这个mask_class_onehot_for_d的shape也是(4, 512, 512, 7)
                labels = tf.reshape(labels, (-1, FLAGS.num_classes))  # 此时labels的shape=(1048576, 7)
                """
                上面三句，reshape之前，labels的shape是(4, 512, 512, 7)，就是先把编码标签mask_class变成1热标签
                    （增加了一维，每个像素点都弄成7个维度的向量，表示该点属于某个类别的概率），然后重排成一列；
                那个mask_class_onehot_for_d，好像和labels差不多，shape也是(4, 512, 512, 7)，只不过这个是给那个D网络用的，所以多了个d啊。
                    但是为啥有个on_value什么的？
                    →→→其实很简单，到时候在定义了sess之后，加个断点run掉，就可以看某个维度上的数值了，这就是不需要feed的好处啊！。。
                       实地看了下，发现加了那个on_value和off_value之后，所有数值都被限制在这两个值之间。
                           后面sess.run那句设断点停下，然后：
                           l1, l2=sess.run([labels,mask_class_onehot_for_d])
                           l1.shape
                           l10=l1[0,:]
                           l10
                           l2.shape
                           l20=l2[0,0,0,:]
                           l20
                           发现l10是[ 1., 0., 0., 0., 0., 0., 0.]，l20是[ 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                           相当于是，把那些绝对的、全都是非0即1的1热标签，换成了0.01~0.99之间的，这样应该可以避免发生nan一类的东西。
                """

                #input_d = tf.reshape(mask_class_onehot_for_d, (-1, FLAGS.num_classes))
               # onedim_class = tf.reshape(mask_class, (-1,))

            #image = preprocessing.data_augmentation(image, is_training=True)
            
            
        #image,mask_class,mask_instance = preprocessing(image,mask_class,mask_instance,is_train=True,data_format= 'NHWC')
                                                                  
        logits,_,_= SpinePathNet.g_l_net(image, batch_size=FLAGS.batch_size, class_num=FLAGS.num_classes,
                                         reuse=False, is_training=True, scope='g_SpinePathNet')  # shape=(4, 512, 512, 7)
        """
        上句，本来是把训练图像输入进去，然后输出logits。logits的shape是(4, 512, 512, 7)。
        现在想用掩膜RCNN来做这个事儿，掩膜RCNN本来是输出各个类别、外接矩形、掩膜。我是可以把类别+掩膜来弄成上面的(4, 512, 512, 7)形式。
        """

        #Gan simultanously descriminate the input is the segmentation predictions or ground truth.        
        D_logit_real = SpinePathNet.d_net(mask_class_onehot_for_d,  class_num = FLAGS.num_classes, reuse=None, is_training=True, scope='d_gan')  # shape=(4, 8, 8, 1)
        D_logit_fake = SpinePathNet.d_net(logits,  class_num = FLAGS.num_classes, reuse=True, is_training=True, scope='d_gan')  # shape=(4, 8, 8, 1)
        """
        以上两句，用GAN网络同时确定输入是分割的预测，还是金标准。
        这两句就是第一个输入不一样啊：
            第一句输入是mask_class_onehot_for_d，这是掩膜的金标准，然后这里作为“输入的真实样本”，就是那个y。
            第二句输入就是logits，这是生成网络弄出来的假数据，就是那个S(x)。
        输出的：
            D_logit_real应该是161中9式中的D(y_{n})，输入为真样本y的时候，判别网络D认为它是真样本的概率D(y)；
            D_logit_fake应该是161中9式中的D(S(x_{n}))，输入为一个假样本S(x)的时候，判别网络D认为它是假样本的概率1-D(G(x))。
            不过，为啥shape是(4,8,8,1)啊，为啥我觉得似乎应该是(4,)？？？
            →→→暂时这样理解：如果那个D_logit_real是一个数，那么，训练D网络的时候，金标准就是1；
                而，现在D_logit_real是一个8*8*1的矩阵（就相当于是8*8的矩阵），那么，金标准就成了8*8个1组成的矩阵。
                这样似乎能够勉勉强强地解释，但是，我总觉得还是很牵强。
                后来又看了两个网站：似乎果然也可以输出一个矩阵（就对应我们的8*8的矩阵）作为D网络的输出，然后取平均值的。那两个网址是：
                    https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/406_GAN.py （代码第51行）
                    和
                    https://www.jianshu.com/p/08abd788d598 （他说discriminator(x)接受MNIST图片作为输入，返回一个代表真实图片概率的张量。）
                不过更诡异的是，run掉那个D_logit_real之后发现，里面居然有数是大于1的或者小于0的，好诡异啊。
                然后把这个矩阵里的8*8个数平均了一下，倒是处于0~1之间的。
                    【其实这么一想，确实可以把“D网络输出的矩阵，逐元素求了平均之后的东西”叫做那个概率D(y)，而不管那个D_logit_real是什么东西了】
                但是那么多个8*8矩阵呢，你确定每个都在0~1之间吗？？？。。。
                    这个做法有点诡异，本来是一张图吧，然后卷积了几次（就是加权平均了几次），就得到一个概率了。
                    不过……想想也有可能，因为无论如何这个概率都是从输入图像来的，对输入图像做了卷积、正则化啊、池化啊这些事儿，
                    然后最后那个平均，不就是相当于是一个8*8的卷积，不要padding，然后卷积核里每个数都是1/64的嘛。所以他输出是个矩阵和是个标量值并没啥特大的区别啊。
            →→→后来我发现论文中图5的caption里说，这个D_logit_real/fake就应该是一个数，那么上面的东西就更有问题了。
                还是要加个卷积层把8*8的变成一个数。庞树茂说，这就有两种方法：
                    1、直接用一个8*8的卷积核去做无padding卷积，这样得到的东西的维度就是8-8+1=1维的，这就相当于是一个全连接了；
                    2、用两次卷积：第一次用5*5的卷积核，变成8-5+1=4维的，第二次再用4*4卷积核。
                    这个是没有一定之规的，只要效果好就可以了。
                重新看了下上面网址里的代码：
                    第一个代码，49行的那句应该是先把输入的D_l1和一个随机的权重矩阵W相乘，再加上个随机的偏置，然后用sigmoid。
                    然后这一行的输出，即prob_artist1的shape应该是(None,1)，None就代表这一批次里的图片数。然后51行的平均应该是对所有图片求平均的。
                    这样看来，对每一张图的判断，D网络还是输出的一个数（而不是矩阵的）。
                    第二个代码，看看他前面怎么matmul的，很明显就能看出来D_prob的shape是(None,1)。所以后面的平均也是对所有图片求平均的。
        这儿也可以看他们弄的是逐像素点的分割，而我们弄的是掩膜。
        """
        
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy_loss = losses.weighted_cross_entropy_with_logits(FLAGS, Fold, logits, labels)
            """
            上句是交叉熵损失，应该是类别判断的损失。就是logits和labels在对比了。前者shape=(4, 512, 512, 7)，后者shape=(1048576, 7)
            这个对应论文中的10式第一项，分割损失。也是13/14式中的第一项。
            看了一下那个函数，就是每张图每个像素点的交叉熵。
            """
            
        with tf.name_scope('gan_loss'): # GAN. GAN网络损失
            """
            【这一段的总体思路】
            损失函数有两部分，一个是生成网络G损失、一个是辨别网络D损失。
                其中G损失有两项：分类标签损失（上面的交叉熵损失）和分割损失（下面的G_loss_fake）
                D损失也是两项：161论文中9式的那两项，9式是先通过调节D网络最大化判别正确的概率，然后通过调节G网络最小化判别正确的概率。
                    不过奇怪的是，不知道为什么明明是先最大化再最小化，而损失函数还是可以这么弄呢？？？
                    →→→看原文P20，从9式变成10式的过程。就是先加负号把最大最小问题变成纯最小问题，然后再把分割损失（程序里的cross_entropy_loss）加上去的。
                一共四项合起来是总损失。不太确定为啥10式中只有三项，而这里变成了四项？？？
                →→→因为10式中第三项既和G网络有关，又和D网络有关，而这两个网络是分别优化的，所以每次优化的时候就都要有这个第三项了。
                    然后优化G网络的时候，第三项做了些改变，就成了那个G_loss_fake（10式第一项、14式第二项）。这个再加上分割损失（14式第一项）。看原文2.4.1节的注释。
                    优化D网络的时候，就是D_loss_real（15式第一项、10式第二项）和D_loss_fake（15式第二项、10式第三项）。
            """
            D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))  # shape=()
            """
            上句对应于161中9式的第一项（或者10式中\mathcal{L}_{bcl}(D(y_{n}),1)一项）：输入为真实样本y的时候，判别网络D认为它是真实样本的概率D(y)，
                然后对所有图取平均（10式中对n求平均）。当然这个地方的平均，不但是对那batch_size(4)张图求平均，而且也对每张图的8*8矩阵做了平均。
            注意那个logits对每张图求平均后，就是式中的D(y)了，而此处的金标准labels就是1（因为输入是金标准啊）。
            """
            D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))  # shape=()
            """
            上句是161中9式的第二项（或者10式中\mathcal{L}_{bcl}(D(S(x_{n})),0)一项）：输入为一个假样本的时候，判别网络D认为它是真实样本的概率1-D(G(x))，
                然后取平均。
            注意那个logits就是式中的D(G(x))了，而此处的金标准labels就是0（因为输入是假数据啊，现在logits是D(G(x))而不是1-D(G(x))）。
            """
            G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))  # shape=()
            """
            这似乎是10式中的\mathcal{L}_{mcl}(S(x_{n}),y_{n})一项。但为啥那个labels是一大堆的1呢？？？
            →→→这个就是14中的第二项。看原文2.4.1节的注释。
            """
            #lmada = tf.Variable(2., name = 'weight_of_gan')
            
            D_loss = D_loss_real + D_loss_fake
            G_loss = cross_entropy_loss + G_loss_fake
            """加和求总的生成网络G损失、辨别网络D损失"""
        print('nets build...')


        # trainable varibles of generative and discrimative models. 生成和判别模型的可训练参数。
        """【学习】选取所有可训练参数、名字中带有某些字符的可训练参数（其实就是某个网络的参数）的方法"""
        t_vars = tf.trainable_variables()  # 所有可训练的参数，是一个list，列出了一大堆的变量名、shape、dtype等等。
        d_vars = [var for var in t_vars if 'd_' in var.name]  # 参数名中有d_的那些可训练参数。其实就是D网络的参数。
        g_vars = [var for var in t_vars if 'g_' in var.name]  # 类似地，G网络的参数。
       # print (g_vars)  # 可以打印出来看看啊。。
        
        learning_rate = tf_utils.configure_learning_rate(FLAGS,FLAGS.num_samples, global_step)  # 应该是当前时代的学习率。
        optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)  # 【这个是把所有的优化子、学习率等都放在tf_utils文件里了】
        optimizer_gan = tf.train.AdamOptimizer(beta1=0.5, learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 从一个结合中取出全部变量，得到一个列表
        print('optimizer set...')
        
        #Note: when training, the moving_mean and moving_variance need to be updated. 训练中更新滑动均值和滑动方差。
        with tf.control_dependencies(update_ops):
            train_op_G = optimizer.minimize(G_loss, global_step=global_step, var_list=g_vars)            
            #train_op_fake = optimizer.minimize(G_loss_fake, var_list=g_vars)            
            train_op_D = optimizer_gan.minimize(D_loss, global_step=global_step, var_list=d_vars)
        """上面这个with似乎就是弄了两个训练op，去最小化生成网络G损失和辨别网络D损失。"""
            
        # The op for initializing the variables. 初始化各种变量。
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Create a session for running operations in the Graph.
        #config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session()
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        print('initialized...')  # 从上面的print('optimizer set...')到此处，至少要执行8~10分钟的时间。
        
        
        #Include max_to_keep=5
        saver = tf.train.Saver()  # 保存训练节点
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)  # 【学习】如果输入的路径中有ckpt文件（训练节点），那么，读取它。
        """【基础】读取以前训练的节点
            ckpt是：
                model_checkpoint_path: "tmp/tfmodels_gan_lstm_5/model.ckpt-12687"
                all_model_checkpoint_paths: "tmp/tfmodels_gan_lstm_5/model.ckpt-9000"
                all_model_checkpoint_paths: "tmp/tfmodels_gan_lstm_5/model.ckpt-10000"
                all_model_checkpoint_paths: "tmp/tfmodels_gan_lstm_5/model.ckpt-11000"
                all_model_checkpoint_paths: "tmp/tfmodels_gan_lstm_5/model.ckpt-12000"
                all_model_checkpoint_paths: "tmp/tfmodels_gan_lstm_5/model.ckpt-12687"
            所以，ckpt.model_checkpoint_path是：
                'tmp/tfmodels_gan_lstm_5/model.ckpt-12687'
        """

        if ckpt and ckpt.model_checkpoint_path:
            exclude_list = FLAGS.ignore_missing_vars
            variables_list  = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
            restore = tf.train.Saver(variables_list)
            restore.restore(sess, ckpt.model_checkpoint_path)
            """
            上面三句是，得到要恢复的变量的列表（除了exclude_list中的不要）之外，然后用tf.train.Saver、restore等命令恢复这些变量。
            第二遍看程序的时候，就应该知道：
            1、如何恢复的：用的是model.ckpt-12687，应该就是model.ckpt-12687.data-00000-of-00001、model.ckpt-12687.index、model.ckpt-12687.meta这三个一起作用，
                才是这个模型里的所有变量信息吧。。
            2、variables_list就是“各个卷积层、全连接层的权重和偏置”的名称、shape和dtype。不过现在有点不放心的是，如果名字不一样但是形状一样，可以恢复吗？？
                并且，我记得如果不一样的话，应该是可以恢复一部分的啊。。
            3、可以先run掉那个init_op再加载变量，这样就不会有的变量忘了初始化了。
            """
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = int(global_step)
            """上面两句，读取已经训练了的全局步数"""
        else: global_step = 0    
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 线程什么的。
        
        
        # Save models.
        if not tf.gfile.Exists(FLAGS.train_dir):  # 保存模型的目录
                #tf.gfile.DeleteRecursively(FLAGS.train_dir)
                tf.gfile.MakeDirs(FLAGS.train_dir)
                
                
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        print('ready to optimize...')  # 从上面的print('initialized...')到此处，倒是挺快的。
        
        try:
            step = global_step # TODO: Continue to train the model, if the checkpoints are exist.
            print ('开始时，已经训练了%d步。'%step)
            while not coord.should_stop():
                start_time = time.time()

                _, dl, dlr, dlf = sess.run([train_op_D, D_loss, D_loss_real, D_loss_fake])
                _,gl, cel, fake_loss,  lr = sess.run([train_op_G, G_loss, cross_entropy_loss, G_loss_fake, learning_rate])
                duration = time.time() - start_time
                """
                弄了一个train_op_G和一个train_op_D，然后sess.run两次，第一次是run掉train_op_D，最小化判别损失，第二次是run掉train_op_G，最小化生成损失。
                
                【基础】在此处可以用matlibplot画出来一张图！
                sess.run那句设断点停下，然后：
                    im=sess.run(image)
                    im0=im[0, :, :, 0]   这得到那个批量中的4张图中的第0张，然后最后一维要给他设成0，否则报错。
                    import matplotlib.pyplot as plt
                    plt.imshow(im0)
                至于那个logits的图片怎么存、怎么可视化，应该是在eval_gan_lstm.py里做的事儿。。
                """
                
                if step % 10 == 0:
                    print('Step %d: All generative loss = %.2f (Cross entropy loss = %.2f, Fake loss of generatation = %.2f); All discrimator loss = %.2f (Discrimator loss of real = %.2f, Discrimator loss of fake = %.2f); Learning rate = %.4f (%.3f sec)' % (step, gl, cel, fake_loss, dl, dlr, dlf, lr, duration))
                    # gl(G_loss)：总生成损失、  cel(cross_entropy_loss)：交叉熵损失、  fake_loss(G_loss_fake)：生成网络损失
                    # dl(D_loss)：总辨别损失、  dlr(D_loss_real)：真样本的辨别损失、    dlf(D_loss_fake)：假样本的辨别损失。
                step += 1
                
                if step % 1000 == 0: #or (step + 1) == FLAGS.max_steps
                    #Increase Gan_loss.
                    #lmada.assign_add(0.1)
                    saver.save(sess, checkpoint_path, global_step=step)
                    ##Add the model evaluatation  in the future.
                                                                      
        except tf.errors.OutOfRangeError:  # 似乎是，如果发生了OutOfRangeError，那么无论如何把已经训练了的步数存下来。
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))  
            saver.save(sess, checkpoint_path, global_step=step) 
            ##Add the model evaluatation  in the future.
            print('Model is saved as %s') % checkpoint_path 
            
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()                                                      
                                                                         

if __name__ == '__main__':
    tf.app.run()    
