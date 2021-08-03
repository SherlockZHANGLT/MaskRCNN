#coding=utf-8
"""
滑脱数据集，基于matterport的数据集。
作者：赵屾。修改于180820。
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import skimage.color
import skimage.io

############################################################
#                      定义数据集                           #
############################################################
class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self._epochs_completed = 0  # 当前训练时代，初始化为0，不知道行不行，如果不行的话，得在config.py里的__init__函数里添加。。
        self._index_in_epoch = 0  # 当前批次开始的索引号
        self._image_ids_shuffle = []  # 用来记录每个时代内，被打乱了的图像序号。

    def add_class(self, source, class_id, class_name):
        """好像是把新的类别（来源、编号、名称）加到那个class_info里去"""
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,  # 好像都是organs
            "id": class_id,  # 是从1开始的整数
            "name": class_name,  # "brainstem","chiasm"那些东西，在organs_training的classes里。
        })

    def add_image(self, source, patient_id, image_id, img, mask, masks, width, height, category, **kwargs):
        """把新的图片（来源、编号、名称等等8个东西）加到那个class_info里去
        类似于add_class函数，也是加上好多字符串下标的数组。"""
        image_info = {
            "patient": patient_id,
            "id": image_id,
            "source": source,
            "image": img,
            "mask": mask,  # 这个是原始的掩膜，是(512, 512)的
            "masks": masks,  # 这个其实是按照每一类分好的掩膜，是(512, 512, N)的
            "width": width,
            "height": height,
            "category": category,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
    def add_image_spondy(self, source, patient_id, label, img, mask, masks, width, height, category, **kwargs):
        """把新的图片（来源、编号、名称等等8个东西）加到那个class_info里去
        类似于add_class函数，也是加上好多字符串下标的数组。

        用来做滑脱的那个，数据集略有不同。
        """
        image_info = {
            "patient": patient_id,
            "label": label,  # 这儿改成了这个病人的滑脱分级情况
            "source": source,  # 这儿改成了是CT还是MRI。
            "image": img,
            "mask": mask,  # 这个是原始的掩膜，是(512, 512)的
            "masks": masks,  # 这个其实是按照每一类分好的掩膜，是(512, 512, N)的
            "width": width,
            "height": height,
            "category": category,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        准备数据的类别，其实就是填充self.source_class_ids（类初始化的时候是空集，执行完了应该填入各种类别了）。
        具体过程未细看。

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        # 上句，self.class_info是个list：[{'id': 0, 'name': 'BG', 'source': ''},
        #     {'id': 1, 'name': 'brainstem', 'source': 'organs'},...
        #     {'id': 6, 'name': 'opticnerveR', 'source': 'organs'}]。
        # 所以self.num_classes是7。
        self.class_ids = np.arange(self.num_classes)
        # 上句，因为刚才self.num_classes是7，所以现在输出是array([0, 1, 2, 3, 4, 5, 6])。
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        # 上句，输出的是['BG', 'brainstem', ..., 'opticnerveR']这个list。
        self.num_images = len(self.image_info)
        # 输出196，self.image_info里面有196个元素，就是说有196张图。
        self._image_ids = np.arange(self.num_images)  # 0~195这196个数。
        self._image_ids_shuffle = self._image_ids  # 一开始的时候，跟那个self._image_ids一样。

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        # 上句，输出{'.0': 0,  'organs.1': 1,  'organs.2': 2,  'organs.3': 3,  'organs.4': 4,  'organs.5': 5,  'organs.6': 6}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        # 上句，输出['', 'organs']。这可能是因为那个self.class_info里的'source'项要么是''，要么是'organs'。
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:  # 刚刚进入此for循环的时候，source=''；执行完了一次下面的小for循环回来的时候，source='organs'。
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset 找到属于这个数据集的类别。
            for i, info in enumerate(self.class_info):
                # 刚刚进入此for循环的时候，i=0、info={'id': 0, 'name': 'BG', 'source': ''}（即self.class_info里的第0个）；
                #     第二次进入的时候，i=1、info={'id': 1, 'name': 'brainstem', 'source': 'organs'}（即self.class_info里的第1个）；
                #     第三次进入的时候，i=2，info={...'source': 'organs'}；...
                #     第六次进入的时候，i=6，info={...'source': 'organs'}；
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
                    # 第一次执行，source=''，i=0，所以是self.source_class_ids['']里面放了个0；
                    # 第二次，i=1但source=''而info['source']是'organs'，所以不执行append；
                    # 第三次，类似地，仍然不执行，直到第六次；
                    # 然后回到上面的for循环，source='organs'，这种情况下无论i=几都执行了append，都加入那个i。
                    # 所以，最后self.source_class_ids['']就是[0]；而self.source_class_ids['organs']就是[0,1,2,3,4,5,6]。
                    # PS：回看前面，这个0~6应该分别代表'BG'、'brainstem'、...'opticnerveR'这些东西。

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        加载选定的图像，返回一个[H,W,3]的np.array（应该就是图像吧）
        """
        # print("load_image::image_id =", image_id)
        # Load image
        # print("Image file =", self.image_info[image_id]['image'])  # 可注释掉
        image = self.image_info[image_id]['image']
        image = np.asarray(image, dtype=float)  # convert to float normalize the image data to 0 - 1
        image = (image / (np.max(image)))
        image = image * 255
        image = image.astype(np.uint8)  # 这儿好像是，图像里的数据都是0~255之间，uint8类型的。
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        # iar = np.asarray(image)  # 可注释掉
        # plt.imshow(image)  # 可注释掉
        # print(iar)  # 可注释掉
        # plt.show()  # 可注释掉

        return image

    def load_mask(self, image_id):
        """
        整合了一下，把原来自己数据集里的load_mask函数整合到这个数据集里去了。。
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        # read mask images
        num_masks = len(image_info["masks"])
        for m in range(0, num_masks):
            mask = image_info["masks"][m]
            # Some objects are so small that they're less than 0.5 pixel area
            # and end up rounded out. Skip those objects.
            if mask.max() < 0.5:
                continue
            instance_masks.append(mask)

        # retrieve saved image info in the variable organs
        organ_class = []
        for i in range(len(image_info["category"])):
            current_class = self.class_names.index(image_info["category"][i])
            organ_class.append(current_class)
        image_info["category_ids"] = organ_class

        # Map class IDs to images
        class_ids = np.array(image_info["category_ids"])
        class_ids = class_ids[:, np.newaxis]
        # 上句，原来那个class_ids是array([[1]])，而我这个是array([1,2,3,4,5,6])，升一维和原来的匹配，弄成array([[1],[3],[4]])的样子

        # Pack instance masks into an array
        if class_ids.all():
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            class_ids = class_ids.reshape(class_ids.shape[0])
            return mask, class_ids
        else:
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
            print("第%d张图，掩膜是空的！" % class_ids)
            return mask, class_ids

    def randomize_img_ids(self):
        """ Randomizes the image ids in the dataset by taking the total number
        of images and generates a random number sequence with np.random.seed()

        N_img = the total number of images in the dataset

        random.permutation(range(N_img)) randomly shuffles the image ids within
        the range of the dataset

        img_dataset stores the new randomized dataset
        """
        np.random.seed()
        N_img = len(self.image_ids)
        rand_img_ids = list(np.random.permutation(range(N_img)))
        img_dataset = [self.image_info[i] for i in rand_img_ids]

        return [rand_img_ids, img_dataset, N_img]