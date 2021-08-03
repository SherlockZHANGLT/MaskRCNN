#coding=utf-8
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline. 主函数：把数据集编程TFRecords格式，这可以很容易地弄到TensorFlow管线中去。
Usage:
下面是用法，大概能看出来这个函数的tf.app.flags的东西，在那个终端（所谓shell）里该怎么用（每个flag都是用一个--符号加上想输入的东西）。。
```shell
python tf_convert_dicom_to_tfrecord.py \
    --dataset_name=spine_segmentation \
    --dataset_dir=./datasets/spine_segmentation/spine_segmentation_5/test/ \
    --output_name=spine_segmentation_test_5_fold \
    --output_dir=./datasets/tfrecords_spine_segmentation_with_superpixels
```
除了在终端中用之外，如果在pycharm里，应该是直接改那个tf.app.flags里面的参数。
"""
import tensorflow as tf

from datasets import convert_dicom_to_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation',  # 这个是不能动的
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', 'E:/赵屾的文件/55-脊柱滑脱/Spine-Gan_李天洋/datasets/spine_segmentation/',  # 注意路径的话，必须最后有个/
    # 大电脑Win系统是E:/赵屾的文件/55-脊柱滑脱/Spine-Gan_李天洋/datasets/spine_segmentation/；Ubuntu系统是/home/zs/PycharmProjects/14_Spine_Gan/datasets/spine_segmentation/
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'spine_segmentation_test_try1',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', 'E:/1',  # 此路径不能由中文字符。先弄到这里再移动吧。发现读是可以，写是写不了。。。E:/赵屾的文件/55-脊柱滑脱/Spine-Gan_李天洋/datasets/spine_segmentation/test/1/
    # 大电脑Win系统是E:/1；小电脑Ubuntu系统是/home/zs/PycharmProjects/14_Spine_Gan/outputs
    'Output directory where to store TFRecords files.')

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'spine_segmentation':  # 这意思是说前面那个dataset_name是不能动的啊。。
        convert_dicom_to_tfrecord.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)
        
        
if __name__ == '__main__':
    tf.app.run()