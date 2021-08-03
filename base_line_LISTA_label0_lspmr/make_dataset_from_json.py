#coding=utf-8
"""
作者：赵屾，20180614。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from skimage import measure
classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]  # 这个是我的检测的json。

def norm_image(image, max_wanted, min_wanted):
    """只搞一张灰度图，输入应该是个二维矩阵。从dicom里来的东西，也确实是二维矩阵的。"""
    image_max = np.max(image)
    image_min = np.min(image)
    image_normed = (max_wanted - min_wanted) * (image - image_min) / (image_max - image_min) + min_wanted
    return image_normed

import class_detection_dataset
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