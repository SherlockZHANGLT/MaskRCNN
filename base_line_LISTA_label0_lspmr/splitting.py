import tensorflow as tf
import numpy as np
import json
import make_dataset_from_json
import class_detection_dataset
"""
本程序做两件事：
1、从输入的（总的）数据集中，分出来CT的或者MRI的；
2、从输入的（或者是总的，或者是CT的，或者是MRI的）数据集中，分出来训练集和测试集。

比起“E:\赵屾的文件\55-脊柱滑脱\Huatuo\make_spondylolisthesis_dataset_python\make_MRCNN_json_from_png.py”里的那个，
    所有的add_image改成了add_image_spondy，其他都一样。
"""
classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]

def get_train_test_sets_from_dataset(dataset, perception_in_testing, mode):
    """本函数中的train应该是训练集和验证集的总共个数，然后具体训练的时候，再把训练集和验证集分开吧。。
    dataset_test应该是不用于训练的。
    dataset_train里是用来训练的，然后，如果要做5折交叉验证的话，那么要分5次，弄成训练集和验证集什么的。
    输出class_detection_dataset类型变量的版本。
    感觉这儿就有个教训，任何的数据集，都不能直接用那个list或者别的类型，而应该子集有一个类型，比如class_detection_dataset类。
        否则的话从数据集弄成batch的时候不好搞啊。。
    """
    assert mode in ['random', 'split1', 'split2', 'split3', 'split4', 'split5']
    dataset_train_spondy = class_detection_dataset.Dataset()  # 构造一个class_detection_dataset类型变量dataset_train_spondy。
    dataset_test_spondy = class_detection_dataset.Dataset()  # 再构造一个class_detection_dataset类型变量dataset_test_spondy。
    assert isinstance(dataset, class_detection_dataset.Dataset), "必须是class_detection_dataset类型变量。"
    all_patients = len(dataset.image_info)
    all_ids = np.arange(all_patients)
    if mode == 'random':
        test_number = int(all_patients / perception_in_testing)
        train_number = all_patients - test_number  # 训练集和验证集的总共个数
        np.random.shuffle(all_ids)  # 打乱顺序
        ids_train = all_ids[:train_number]  # 训练集和验证集的样例索引号
        ids_test = all_ids[train_number:]  # 测试集的样例索引号。
        """注意，上面两句，ids_train和ids_test合起来应该是0~134，
        即，此时的ids_train里的数就是原来的病人序号（即那个dcm文件的编号）。
        """
    elif mode == 'split1':
        ids_test = [200, 202, 203, 205, 210, 213, 214, 217, 219, 221, 223, 224, 225, 227, 229,
                    266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
                    341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
                    198, 201, 204, 206, 207, 208, 209, 211, 212, 216, 218, 220, 222, 226, 228,
                    416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,
                    435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,
                    454, 463, 465, 469, 474,
                    479, 491, 492, 498, 499,
                    507, 511, 512, 521, 524,
                    532, 539, 542, 547, 548,
                    558, 559, 565, 571, 572,
                    579, 584, 589, 598, 599]
        # ids_test = [i - 1 for i in ids_test]  # 因为dataset.image_info[i]里是从0~449的，但是那个索引号是从1到450的，所以要都减1。
        # --后来加入癌症数据集的时候，把上面的索引号都手动减了1，就不用了。下同。
        # 格式是，上面6行是原来师妹数据集的、6种FOV中、测试集中的病人，下面6行是新癌症数据集的、6种FOV中、测试集中的病人。
        ids_train = list(set(all_ids.tolist()).difference(set(ids_test)))  # 先变成list，再变成list组成的set，求差集，再变成list。
    elif mode == 'split2':
        ids_test = [119, 122, 124, 125, 128, 131, 134, 137, 148, 158, 170, 173, 185, 188, 194,
                    251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
                    326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                    166, 167, 168, 171, 174, 176, 177, 179, 184, 190, 191, 192, 193, 195, 196,
                    401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
                    178, 180, 181, 182, 183, 186, 187, 189, 197, 199, 215, 431, 432, 433, 434,
                    453, 458, 459, 464, 473,
                    478, 486, 487, 496, 497,
                    503, 510, 516, 522, 523,
                    530, 533, 537, 538, 549,
                    553, 556, 563, 564, 569,
                    578, 583, 588, 593, 594]
        ids_train = list(set(all_ids.tolist()).difference(set(ids_test)))  # 求差集。
    elif mode == 'split3':
        ids_test = [63, 66, 69, 74, 77, 80, 82, 85, 98, 100, 101, 103, 104, 106, 115,
                    236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                    311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
                    145, 146, 147, 149, 150, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162,
                    386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400,
                    123, 126, 129, 132, 135, 136, 138, 141, 151, 163, 164, 166, 169, 172, 175,
                    452, 457, 462, 468, 472,
                    477, 482, 483, 488, 493,
                    502, 504, 508, 514, 515,
                    529, 531, 534, 543, 546,
                    552, 557, 560, 566, 573,
                    577, 582, 587, 592, 597]
        ids_train = list(set(all_ids.tolist()).difference(set(ids_test)))  # 求差集。
    elif mode == 'split4':
        ids_test = [28, 30, 31, 33, 37, 38, 43, 46, 47, 54, 55, 57, 59, 60, 61,
                    65, 68, 71, 72, 73, 75, 76, 78, 79, 230, 231, 232, 233, 234, 235,
                    296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                    97, 107, 109, 110, 113, 116, 117, 127, 130, 133, 139, 140, 142, 143, 144,
                    371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
                    67, 70, 91, 92, 96, 99, 102, 105, 108, 111, 112, 114, 118, 120, 121,
                    451, 456, 461, 467, 471,
                    476, 480, 481, 489, 494,
                    501, 506, 513, 517, 520,
                    527, 528, 535, 541, 544,
                    551, 554, 562, 568, 574,
                    576, 581, 586, 591, 596]
        ids_train = list(set(all_ids.tolist()).difference(set(ids_test)))  # 求差集。
    else:
        ids_test = [1, 2, 5, 7, 10, 13, 16, 17, 19, 20, 23, 24, 25, 26, 27,
                    0, 3, 4, 6, 8, 9, 11, 22, 29, 34, 35, 49, 50, 62, 64,
                    281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
                    14, 40, 41, 53, 81, 83, 84, 86, 87, 88, 89, 90, 93, 94, 95,
                    356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370,
                    12, 15, 18, 21, 32, 36, 39, 42, 44, 45, 48, 51, 52, 56, 58,
                    450, 455, 460, 466, 470,
                    475, 484, 485, 490, 495,
                    500, 505, 509, 518, 519,
                    525, 526, 536, 540, 545,
                    550, 555, 561, 567, 570,
                    575, 580, 585, 590, 595]
        ids_train = list(set(all_ids.tolist()).difference(set(ids_test)))  # 求差集。
    # 以上split1~5是个五折交叉验证的。
    for i, classname in enumerate(classes):
        dataset_train_spondy.add_class("organs", i + 1, classname)
        dataset_test_spondy.add_class("organs", i + 1, classname)  # 类似于spondylolisthesis_dataset类型变量的初始化吧。
        # 反正就是类似于make_MRCNN_json_from_png中的get_MRCNN_json函数吧，把dataset里的每个成员变量都弄到。
    for i in range(all_patients):
        # 输入的dataset是spondylolisthesis_dataset类型变量，现在是其中第i张图，把各个成员变量都拆出来。
        patient_id = dataset.image_info[i]['patient']
        image = dataset.image_info[i]['image']
        mask = dataset.image_info[i]['mask']
        masks = dataset.image_info[i]['masks']
        width = dataset.image_info[i]['height']
        height = dataset.image_info[i]['width']
        category_for_dataset = dataset.image_info[i]['category']
        if i in ids_train:
            dataset_train_spondy.add_image(patient_id, image, mask, masks, width, height, category_for_dataset)
        else:  # if i in ids_test
            dataset_test_spondy.add_image(patient_id, image, mask, masks, width, height, category_for_dataset)
    dataset_train_spondy.prepare()
    dataset_test_spondy.prepare()
    return dataset_train_spondy, dataset_test_spondy

def main(_):
    json_dir = 'E:/赵屾的文件/59-脊柱检测/程序/MaskRCNN_and_CRF/detection_ver190129.json'
    #
    file = open(json_dir, 'r', encoding='utf-8')
    s = json.load(file)
    file.close()
    print(len(s))
    print(len(s['images']))  # 现在s里应该有所有图（i=0~134）的图像和掩膜。
    dataset = make_dataset_from_json.get_MRCNN_json(s)
    dataset_train_MRI1, dataset_test_MRI1 = get_train_test_sets_from_dataset(dataset, 5, 'split1')
    print('自己看一下数据集对不对。')

if __name__ == '__main__':
    tf.app.run()