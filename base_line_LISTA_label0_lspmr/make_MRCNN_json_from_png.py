#coding=utf-8
"""
本函数把最原始的png图像弄成dataset，然后保存为json形式。
就是一开始弄那个json文件的时候用的，正式执行的时候是不需要这个函数的。
"""
import tensorflow.compat.v1 as tf
import argparse
import class_detection_dataset
import numpy as np
import os
from PIL import Image
import scipy.ndimage
from skimage import measure
import json
DIRECTORY_ANNOTATIONS = 'all_gt_masks_png/ver_3(dataset)/'
DIRECTORY_IMAGES = 'all_images_png/ver_3(dataset)/'

np.set_printoptions(threshold=np.inf)

classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]

def get_groundtruth_mask(png_mask, wanted_shape_x, wanted_shape_y):
    M = np.array(png_mask, dtype=np.uint8)
    xscale = wanted_shape_x / M.shape[0]
    yscale = wanted_shape_y / M.shape[1]
    mask_data = scipy.ndimage.interpolation.zoom(M, [xscale, yscale], order=0)
    """【注意】掩膜插值的问题
    对于掩膜来说，如果xscale/yscale不是1，上一句的缩放插值可能会有问题。比如说，如果掩膜是第2类的，他有可能在中间的一大片2的边缘
        插入一些1，这样就影响后面的程序了。。
    解决方法是把插值改成order=0，相当于是最邻近插值（1是线性，2是二次插值，等等），这样不会插出来1，但是对于边缘可能会有影响。
    """
    # print('缩放后每张图的大小：', mask_data.shape)
    return mask_data

def get_image_from_png(ori_image, wanted_shape_x, wanted_shape_y):
    I = np.array(ori_image, dtype=np.uint8)
    xscale = wanted_shape_x / I.shape[0]
    yscale = wanted_shape_y / I.shape[1]
    image_data = scipy.ndimage.interpolation.zoom(I, [xscale, yscale])
    # print('缩放后每张图的大小：', image_data.shape)
    return image_data

def make_MRCNN_json(dataset_dir, wanted_shape_x, wanted_shape_y):
    """
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)  # 那些xml文件的路径名。
    # filenames = sorted(os.listdir(path))  # 所有的xml文件名的列表。这个sort弄的跟傻逼一样，1完了是10，然后是100,101,...199完了才是2，sb的东西。。。
    filenames = []  # 所有的文件名的列表。这个sort弄的跟傻逼一样，1完了是10，然后是100,101,...199完了才是2，sb的东西。。。
    filenames_temp = os.listdir(path)
    while filenames_temp:
        filename = filenames_temp.pop()  # 最后一个文件名。
        filename = filename[:-4]
        try:
            filename = int(filename)
            filenames.append(filename)
        except:
            print ('文件名%s不是数字，没有加入到序列中。' % filename)
    filenames = sorted(filenames)
    images = []
    masks = []
    for patient_id in range(len(filenames)):  #len(filenames)  如果要弄小数据集，把那个len(filenames)改成9（9张图）！！
        print('\r>> 正在把第%d张图（共%d张）加载到数据集中......' % (patient_id, len(filenames)))
        # if patient_id in [108, 323, 383, 384]:
        #     print ('检查')
        img_name = filenames[patient_id]
        ori_image_dir = dataset_dir + DIRECTORY_IMAGES + str(img_name) + '.png'  # 这个没问题。。
        ori_image = np.array(Image.open(ori_image_dir))
        png_mask_dir = dataset_dir + DIRECTORY_ANNOTATIONS + str(img_name) + '.png'
        png_mask = np.array(Image.open(png_mask_dir))
        assert ori_image.shape == png_mask.shape, '原图和掩膜的尺寸必须相同！'
        if ori_image.shape != (512, 512):
            print('\r第%d张图，原图像/掩膜大小不是512*512，已经经过缩放。' % patient_id)
        image = get_image_from_png(ori_image, wanted_shape_x, wanted_shape_y)
        mask = get_groundtruth_mask(png_mask, wanted_shape_x, wanted_shape_y)

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.imshow(image)
        # savename = str(patient_id) + '.png'
        # plt.savefig(savename, bbox_inches="tight", pad_inches=0, ax=8)
        # plt.imshow(mask)
        # savename1 = str(patient_id) + '_mask.png'
        # plt.savefig(savename1, bbox_inches="tight", pad_inches=0, ax=8)

        images.append(image.tolist())  # 每张图都变成list，然后再append。
        masks.append(mask.tolist())
    print('\n把图片读入进来了，等待他存为json文件。')  # 循环完了之后，那个tfrecord文件才会被统一改写（即循环完后，文件大小才会变得不是0）
    return images, masks

def get_MRCNN_json(loaded_json_file):
    s = loaded_json_file
    images = s['images']
    images = np.array(images)  # 刚才是list，现在变成array，应该是(135, 512, 512)的np.array。
    masks = s['masks']
    masks = np.array(masks)  # 应该也是(135, 512, 512)的np.array。

    all_patients = images.shape[0]  # 一共多少张图。
    dataset = class_detection_dataset.Dataset()  # 构造了一个Dataset类型变量dataset。
    for i, classname in enumerate(classes):
        dataset.add_class("organs", i + 1, classname)
    image_info = {}
    width = images.shape[1]
    height = images.shape[2]
    images_for_dataval = []
    for p in range(all_patients):  # all_patients
        patient_id = p
        image_temp = images[p, :, :]  # 这儿只有1张图了。。
        mask_temp = masks[p, :, :]
        image_temp = image_temp.astype(np.uint8)  # 变成np.uint8【此处忘了图像归一化，还好读进来的图就是给归一化到0~255的，否则就会不对了。。】
        mask_temp = mask_temp.astype(np.uint8)  # 变成np.uint8
        non_zero_ix = mask_temp.nonzero()
        non_zero = mask_temp[non_zero_ix]  # 非零的类别序号。
        classes_num = np.max(non_zero) - np.min(non_zero) + 1 + 1  # 这张图一共多少类。最后的+1是加上背景类。

        mask = np.zeros((width, height, classes_num))
        mask_labeled = np.zeros((width, height, classes_num))  # 把每一类的再分成几块。
        all_organs = 0  # 一开始的器官数
        organ_num = np.zeros(classes_num)  # 每种器官的个数
        for i in range(np.min(non_zero), np.max(non_zero)+1):  # i=0的是背景，就让他是一大堆0好了，然后这个i就是从最小的类别标签到最大的类别标签。
            """此for循环，详见后面的注释。。"""
            i1 = i - np.min(non_zero) + 1  # 从label最小的类别开始，这样，如果没有第1类，那么mask[:, :, 1]就是第2类的所有物体的总掩膜。
            mask[:, :, i1] = mask_temp == i  # 三维矩阵的第三维，这儿用[:,:,i]而MATLAB里是(:,:,i)。
            mask_labeled[:, :, i1] = measure.label(mask[:, :, i1], connectivity=1)  # 这一类（第i类）所有物体，每一个物体的掩膜给一个不同的值。
            m = np.max(mask_labeled[:, :, i1])
            all_organs = all_organs + m  # 一共多少个物体
            organ_num[i1] = m  # 这一类一共多少个物体
        all_organs = all_organs.astype(int)
        mask_detailed = np.zeros((width, height, all_organs + 1))  # 要+1。
        category_detailed = np.zeros(all_organs + 1)
        sum_before = int(1)
        for i in range(1, classes_num):  # 对每个类循环
            num = (organ_num[i]).astype(int)  # 这一类一共多少个物体。这儿是i而不是上面的i1了。
            category_detailed[sum_before:sum_before + num] = i + np.min(non_zero) - 1 # 这一类的物体的label。
            for j in range(sum_before, sum_before + num):  # j是在sum_before~sum_before+num-1之间取值
                mask_detailed[:, :, j] = (mask_labeled[:, :, i] == j - sum_before + 1)
                """
                上句，j在sum_before, sum_before + num之间取值，那就是从1到all_organs之间
                同时，j-sum_before+1在1~num之间取值。如果第i种器官有num个，
                    那这num个肯定都在mask_labeled[:,:,i]这张图里，而且每个的掩膜已经被标记为1~num这些数了。
                验证了一下，上面都是没问题的，其中：
                    1、mask[:, :, 1]~mask[:, :, classes_num-1]（classes_num-1通常是6）是每一种器官（通常共6种）的掩膜；
                    2、mask_labeled[:, :, 1]~mask_labeled[:, :, classes_num-1]是和上面的是一样的，也是每一种器官的掩膜，
                        只不过每一种器官中，“同种而不同个”的器官用了不同的数字去表示（如第一种器官Normal Vertebrae可能有4个，
                        在这个掩膜里就表示为1~4）；
                    3、mask_detailed[:,:,1]~mask_detailed[:,:,all_organs]（all_organs通常是22）是每一个器官的掩膜，
                        此处的第三维就是每一个器官，然后第i个掩膜的器官的种类，就对应相应的category_detailed[i]。
                    4、注意到，mask[:, :, 0]、mask_labeled[:, :, 0]、mask_detailed[:,:,0]，所以：
                        有6种器官，那么mask和mask_labeled就都到[:,:,6]；
                        有22个器官，那么category_detailed和mask_detailed就都到22。
                """
            sum_before = sum_before + num
        category_detailed = category_detailed.astype(int)
        masks_for_dataset = []
        category_for_dataset = []
        for j in range(1, all_organs + 1):  # 对所有的器官循环
            # 器官总个数，一般就是22个。j让他从1~22（而不是从0到21），因为category_detailed的索引是从0到22，而category_detailed[0]是背景。
            classname = classes[category_detailed[j] - 1]  # 本来是classes[category_detailed[j] - 1]现在为了避免保存文字，就不要了
            """
            上句，如果category_detailed[j]=1的话，那么对应的应该是Normal Vertebrae，但是呢，
                这个Normal Vertebrae对应的是classes[0]啊。
            """
            masks_for_dataset.append(mask_detailed[:, :, j].astype(int))
            # j=1的时候，是第1个器官，即应该对应mask[:, :, 1]  试过用.astype(int)转成int，结果他输出的是np.int32，不能用，草他妈的。
            category_for_dataset.append(classname)  # 这个应该是'L4', 'L5', 'S1'那三种器官了。
        images_for_dataval.append(image_temp.astype(np.uint8))  # 先变成uint8再添加到images_for_dataval。
        dataset.add_image(patient_id, image_temp, mask_temp, masks_for_dataset, width, height, category_for_dataset)
    image_info["organs"] = dataset.image_info
    dataset.prepare()
    return dataset

def main(_):
    parser=argparse.ArgumentParser(description="Demon of argparse")
    parser.add_argument('--mode',default='use')
    args=parser.parse_args()
    mode = args.mode
    if mode == 'get':
        dataset_dir = 'E:/赵屾的文件/59-脊柱检测/数据/张冉冉的数据/'  #
        print('数据集路径:', dataset_dir)
        wanted_shape_x = 512  # 512  224
        wanted_shape_y = 512  # 512  224
        images, masks = make_MRCNN_json(dataset_dir, wanted_shape_x, wanted_shape_y)  # 已经看了，和MaskRCNN_9_main.py里弄出来的dataset是完全一样的。
        file = open('detection_ver190129.json', 'w', encoding='utf-8')
        d = dict(images=images, masks=masks)
        # 【基础】上句弄了个dict，存完了之后，可以读了。如果直接用那个default=lambda obj: obj.__dict__，就不能读，后面json.load(file)报错，估计是他自己编码弄乱掉了。
        json.dump(d, file, ensure_ascii=False)  # 450个数据，大概15分钟。
        file.close()
        print('已经把数据集转化为json文件。')
    if mode == 'use':
        #json_dir = 'D:/2021/2020下/大创/数据集/t2_1-10/003/t2s_dcm_Label/t2s_dcm_Label.json'
        json_dir = 'D:/2021/2020下/大创/数据集/椎骨检测1/椎骨检测/data/detection_ver200204.json'
        #
        file = open(json_dir, 'r', encoding='utf-8')
        s = json.load(file)
        file.close()
        print (len(s))
        print(len(s['images']))  # 现在s里应该有所有图（i=0~134）的图像和掩膜。
        dataset = get_MRCNN_json(s)
        print('从json文件里弄出来了数据集，并且把掩膜按照不同器官分出来了。')  # 读出来，折腾完这些东西，也就1~2分钟。
        f=open("result.txt",'w+')
        print(dataset.image_info,file=f)
        f.close()
        print('已经把json文件弄成dataset的形式。')

if __name__ == '__main__':
    tf.app.run()