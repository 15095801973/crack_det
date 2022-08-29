import math

import tensorflow as tf
import os
import sys
import random
import json
import datetime
import numpy as np
import skimage.draw

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("")

import cv2

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # 找到本地库
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
# import mrcnn.model as modellib
# from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn.model import log
MODEL_DIR = os.path.join(ROOT_DIR,"logs")
DATASETS_DIR = os.path.join(ROOT_DIR,"datasets/crack_huge")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)

def get_ax(rows=1, cols=1, size=8):
    """返回一个Matplotlib轴数组，用于笔记本中的所有可视化。
    提供一个控制图形大小的中心点。
    更改“默认大小”属性以控制渲染图像的大小
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def rand(a=0, b=1):
    """
    生成一个取值范围为[a,b)
    :param a:
    :param b:
    :return:
    """
    return np.random.rand() * (b - a) + a

class CracksConfig(Config):
    """训练混凝土表面裂缝检测模型时的配置.
    """
    # 给予这个配置一个易于识别的名称
    NAME = "cracks"

    # 在一个GPU上训练,并且每个GPU每次只训练一张图片
    # 为了避免显存不足,尽量往小的方向配置
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 类别数量,实体+背景,这里只考虑混凝土裂缝和背景
    NUM_CLASSES = 1 + 1

    # 使用小尺寸的图片以便训练得更快,训练前将其全部调整为128x128大小
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # 因为数据集是混凝土表面裂缝,往往贯穿整个图片,所以必须
    # 至少设置一个128的anchors才能捕获
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # 每张图片的训练感兴趣区域,不需要太大,训练集里一张图片只有一两条裂缝
    # 至少我标注得是这样,对于一些形状丰富的可能需要几个检测才能满足
    # 能够保证取到正感兴趣区域.
    TRAIN_ROIS_PER_IMAGE = 8

    # 每个epoch训练多少次
    STEPS_PER_EPOCH = 100

    # 每个epoch验证多少次
    VALIDATION_STEPS = 5
class CracksConfig2(Config):
    """训练混凝土表面裂缝检测模型时的配置.
    """
    # 给予这个配置一个易于识别的名称
    NAME = "cracks"

    # 在一个GPU上训练,并且每个GPU每次只训练一张图片
    # 为了避免显存不足,尽量往小的方向配置
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 类别数量,实体+背景,这里只考虑混凝土裂缝和背景
    NUM_CLASSES = 1 + 1

    # 使用小尺寸的图片以便训练得更快,训练前将其全部调整为128x128大小
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # 因为数据集是混凝土表面裂缝,往往贯穿整个图片,所以必须
    # 至少设置一个128的anchors才能捕获
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    USE_MINI_MASK = False
    #False

    # BACKBONE_STRIDES = [8, 16, 32, 64, 128]
    # POOL_SIZE = 14
    MASK_POOL_SIZE = 28
    POOL_SIZE = 7 # 7
    MASK_SHAPE = [56, 56]
    # POST_NMS_ROIS_TRAINING = 200
    # POST_NMS_ROIS_INFERENCE = 100
    # 每张图片的训练感兴趣区域,不需要太大,训练集里一张图片只有一两条裂缝
    # 至少我标注得是这样,对于一些形状丰富的可能需要几个检测才能满足
    # 能够保证取到正感兴趣区域.
    # TRAIN_ROIS_PER_IMAGE = 4
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    # 每个epoch训练多少次
    STEPS_PER_EPOCH = 10

    # 每个epoch验证多少次
    VALIDATION_STEPS = 1

config = CracksConfig2()

# class InferenceConfig(CracksConfig):
class InferenceConfig(CracksConfig2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Crack500_min2_Dataset(utils.Dataset):
    """加载并预处理准备的CRACK500数据集
        """
    def load_balloon(self, dataset_dir, subset):
        self.add_class("crack", 1, "crack")

        # 训练或者是验证
        # assert subset in ["train", "val"]
        dataset_index = os.path.join(dataset_dir, subset)
        lines = []
        for line in open(dataset_index, encoding="utf-8"):
            lines.append(line)
        for i in range(0,len(lines),2):
            tmp = lines[i].split()
            img_path = tmp[0]
            mask_path = tmp[1]
            full_img_path = os.path.join(dataset_dir,img_path)
            full_mask_path = os.path.join(dataset_dir,mask_path)

            image = skimage.io.imread(full_img_path)
            height, width = image.shape[:2]
            print("height, width = ", height, width)
            # 使用文件名作为图片的唯一标识
            self.add_image(
                "crack",
                image_id=img_path+"min2_0",
                path=full_img_path,
                width=width, height=height,
                mask_path = full_mask_path,
                position = 0)
            self.add_image(
                "crack",
                image_id=img_path + "min2_1",
                path=full_img_path,
                width=width, height=height,
                mask_path=full_mask_path,
                position=1)
            self.add_image(
                "crack",
                image_id=img_path + "min2_2",
                path=full_img_path,
                width=width, height=height,
                mask_path=full_mask_path,
                position=2)
            self.add_image(
                "crack",
                image_id=img_path + "min2_3",
                path=full_img_path,
                width=width, height=height,
                mask_path=full_mask_path,
                position=3)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        position = info["position"]
        width, height = info["width"], info["height"]
        min_level = 2

        if position == 0:
            image = image[:height//2,:width//2,:]
        elif position == 1:
            image = image[:height//2,width//2:width,:]
        elif position ==2:
            image = image[height//2:height,:width//2,:]
        elif position ==3:
            image = image[height//2:height,width//2:width,:]


        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "crack":
            return super(self.__class__, self).load_mask(image_id)

        # 转换多边形为位图形式的mask
        # 形状为[height, width, instance_count]
        info = self.image_info[image_id]
        position = info["position"]
        width, height = info["width"], info["height"]
        mask = np.zeros([info["height"]//2, info["width"]//2, 1],
                        dtype=np.uint8)

        org_mask = skimage.io.imread(info["mask_path"])
        # mask[:,:,0] = org_mask[:,:]
        if position == 0:
            mask[:,:,0] = org_mask[:height//2,:width//2]
        elif position == 1:
            mask[:,:,0] = org_mask[:height//2,width//2:width]
        elif position ==2:
            mask[:,:,0] = org_mask[height//2:height,:width//2]
        elif position ==3:
            mask[:,:,0] = org_mask[height//2:height,width//2:width]
        # 返回mask和每个实例的类别标识的数组
        # 因为数据集中只对混凝土裂缝进行了标注
        # 实例类只有一种即裂缝类,所以全部返回1
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)


class Crack500Dataset(utils.Dataset):
    """加载并预处理准备的CRACK500数据集
        """
    def load_balloon(self, dataset_dir, subset):
        self.add_class("crack", 1, "crack")

        # 训练或者是验证
        # assert subset in ["train", "val"]
        dataset_index = os.path.join(dataset_dir, subset)
        lines = []
        for line in open(dataset_index, encoding="utf-8"):
            lines.append(line)
        for i in range(0,len(lines),2):
            tmp = lines[i].split()
            img_path = tmp[0]
            mask_path = tmp[1]
            full_img_path = os.path.join(dataset_dir,img_path)
            full_mask_path = os.path.join(dataset_dir,mask_path)

            image = skimage.io.imread(full_img_path)
            height, width = image.shape[:2]
            print("height, width = ", height, width)
            # 使用文件名作为图片的唯一标识
            self.add_image(
                "crack",
                image_id=img_path,
                path=full_img_path,
                width=width, height=height,
                mask_path = full_mask_path)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "crack":
            return super(self.__class__, self).load_mask(image_id)

        # 转换多边形为位图形式的mask
        # 形状为[height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], 1],
                        dtype=np.uint8)

        org_mask = skimage.io.imread(info["mask_path"])
        mask[:,:,0] = org_mask[:,:]
        # 返回mask和每个实例的类别标识的数组
        # 因为数据集中只对混凝土裂缝进行了标注
        # 实例类只有一种即裂缝类,所以全部返回1
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)


class CrackDataset(utils.Dataset):
    """加载并预处理准备的混凝土裂缝数据集
    """
    def load_balloon(self, dataset_dir, subset):
        """
        加载混凝土裂缝数据集中的一个子集
        :param dataset_dir: Root directory of the dataset.
        :param subset: Subset to load: train or val
        :return:
        """

        # Add classes. We have only one class to add.
        # 添加一个实体类别classes, 我们只需要添加一个裂缝类
        # 背景类别的id为0, 裂缝类别的id为1
        self.add_class("crack", 1, "crack")

        # 训练或者是验证
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 加载标注
        # VGG Image Annotator 保存标注为json文件的格式如下
        # "1.jpg62728": {
        #     "filename": "1.jpg",
        #     "size": 62728,
        #     "regions": [
        #         {
        #             "shape_attributes": {
        #                 "name": "polygon",
        #                 "all_points_x": [...],
        #                 "all_points_y": [...]
        #             },
        #             "region_attributes": {}
        #         }
        #     ],
        #     "file_attributes": {}
        # },
        # 我们需要关注的是regions里面的x和y坐标的值,这表示每个顶点的坐标

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys


        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # VIA工具可能为一个没有标注的图片保存到JSON
        # 毕竟有图片中没有任何实体的情况发生,这种情况也得考虑
        # 但对于本研究来说没有太大价值,所以跳过
        annotations = [a for a in annotations if a['regions']]

        # 添加图片
        print("add images", len(annotations))
        for a in annotations:
            # 首先获取描绘出混凝土裂缝轮廓的多边形的顶点的x,y坐标,
            # 这些坐标的值存储在shape_attributes中,参见上述json格式
            # VIA 1.x版本时:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            # VIA 2.x版本时:
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() 需要图片的尺寸来转换多边形到蒙版
            # 不幸地是, VIA 生成的JSON没有包括这一信息,
            # 所以必须使用读取图片的一种方法来读取图片的尺寸
            # 这种方法只适用于数据集比较小的情况
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            print("height, width = ", height, width)
            print("adding original_img", a['filename'])
            # 使用文件名作为图片的唯一标识
            self.add_image(
                "crack",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

            original_image = Image.open(image_path)
            iw, ih = original_image.size
            print("iw,ih = ", iw, ih)
            FIRST = True
            # 色域扭曲  如果是继续训练的话就不要再生成了,随机的,可能有点影响
            if FIRST:
                hue = rand(-1, 1)
                sat = rand(1, 1.5) if rand()<.5 else 1/rand(1, 1.5)
                val = rand(1, 1.5) if rand()<.5 else 1/rand(1, 1.5)
                # 将图片从RGB图像调整到hsv色域上之后，再对其色域进行扭曲
                x = rgb_to_hsv(np.array(original_image) / 255.)
                x[..., 0] += hue
                x[..., 0][x[..., 0] > 1] -= 1
                x[..., 0][x[..., 0] < 0] += 1
                x[..., 1] *= sat
                x[..., 2] *= val
                x[x > 1] = 1
                x[x < 0] = 0
                COLORED_image = hsv_to_rgb(x)*255  # numpy array, 0 to 1
                print(COLORED_image.shape)
                print("adding COLORED_img")
            COLORED_path = os.path.join(dataset_dir, "COLORED_" + a['filename'])
            if FIRST:
                COLORED_image = Image.fromarray(COLORED_image.astype('uint8'))
                COLORED_image.save(COLORED_path)
            # 使用"COLORED_+文件名"作为图片的唯一标识
            self.add_image(
                "crack",
                image_id="COLORED_" + a['filename'],
                path=COLORED_path,
                width=width, height=height,
                polygons=polygons)

            # 左右翻转图像并另存为
            print("adding FLIP_LEFT_RIGHT_img")
            image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
            FLIP_LR_path = os.path.join(dataset_dir, "FLIP_LEFT_RIGHT_" + a['filename'])
            image.save(FLIP_LR_path)
            # 对于标注的多边形也有同步进行翻转
            polygons_temp = []
            # 遍历一张图里可能会有的多个实体的mask
            for i in range(len(polygons)):
                # 对于每个实体,生成翻转后的新的标注
                dic = {}
                dic['name'] = polygons[i]['name']
                dic['all_points_y'] = polygons[i]['all_points_y']
                # 注意坐标翻转的时候需要-1
                dic['all_points_x'] = [iw - a - 1 for a in polygons[i]['all_points_x']]
                polygons_temp.append(dic)
            # 使用"FLIP_LEFT_RIGHT_+文件名"作为图片的唯一标识
            self.add_image(
                "crack",
                image_id="FLIP_LEFT_RIGHT_" + a['filename'],  # use file name as a unique image id
                path=FLIP_LR_path,
                width=width, height=height,
                polygons=polygons_temp)
            # 上下翻转图像并另存为
            print("adding FLIP_TOP_BOTTOM_img")
            image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
            FLIP_UD_path = os.path.join(dataset_dir, "FLIP_TOP_BOTTOM_" + a['filename'])
            image.save(FLIP_UD_path)
            # 对于标注的多边形也有同步进行翻转
            polygons_temp = []
            # 遍历一张图里可能会有的多个实体的mask
            for i in range(len(polygons)):
                # 对于每个实体,生成翻转后的新的标注
                dic = {}
                dic['name'] = polygons[i]['name']
                dic['all_points_x'] = polygons[i]['all_points_x']
                # 注意坐标翻转的时候需要-1
                dic['all_points_y'] = [ih - a - 1 for a in polygons[i]['all_points_y']]
                polygons_temp.append(dic)
            # 使用"FLIP_TOP_BOTTOM_+文件名"作为图片的唯一标识
            self.add_image(
                "crack",
                image_id="FLIP_TOP_BOTTOM_" + a['filename'],
                path=FLIP_UD_path,
                width=width, height=height,
                polygons=polygons_temp)

    def load_mask(self, image_id):
        """为某个图片生成混凝土裂缝的实例蒙版mask.
       Returns:
        masks: 一个布尔型数组,其形状为[height, width, instance count]
        即每个混凝土裂缝实例的蒙版
        class_ids: masks的类别标识,背景为0,混凝土裂缝为1.
        """
        #         print(self.image_info[image_id])
        # 如果不是混凝土裂缝数据集,返回给父类处理
        image_info = self.image_info[image_id]
        if image_info["source"] != "crack":
            return super(self.__class__, self).load_mask(image_id)

        # 转换多边形为位图形式的mask
        # 形状为[height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # 获取在标注的多边形之内的像素点的索引并设置为1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回mask和每个实例的类别标识的数组
        # 因为数据集中只对混凝土裂缝进行了标注
        # 实例类只有一种,所以全部返回1
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """返回图片的存储路径."""
        info = self.image_info[image_id]
        if info["source"] == "crack":
            return info["path"]
        # 如果不是混凝土裂缝数据集,返回给父类处理
        else:
            super(self.__class__, self).image_reference(image_id)

def prepare_datasets():
    """
    准备数据集并返回
    :return: dataset_train,dataset_val
    """
    # 训练集
    dataset_train = CrackDataset()
    dataset_train.load_balloon(DATASETS_DIR, "train")
    dataset_train.prepare()

    # 验证集
    dataset_val = CrackDataset()
    dataset_val.load_balloon(DATASETS_DIR, "val")
    dataset_val.prepare()
    return dataset_train,dataset_val

def check_dataset():
    dataset_train, dataset_val = prepare_datasets()
    # 加载并随机显示样本
    display_num = 4
    image_ids = np.random.choice(dataset_train.image_ids, display_num)
    for image_id in image_ids:
        #从文件系统中读取图片
        image = dataset_train.load_image(image_id)
        #获取mask和类别标识符
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # 加载随机的图片和mask
    #    dataset = dataset_val
    dataset = dataset_train
    # 随机选取一个id
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    # 从文件系统中读取图片
    image = dataset.load_image(image_id)
    # 获取mask和类别标识符
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    log("previous_img", image)
    # 调整大小
    image, window, scale, padding, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    log("resized_img", image)
    mask = utils.resize_mask(mask, scale, padding)
    # 根据mask计算出边界框,即最大和最小
    bbox = utils.extract_bboxes(mask)

    # 可视化图片和一些状态
    print("image_id: ", image_id, dataset.image_reference(image_id))
    print("Original shape: ", original_shape)
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # 可视化图片和实例
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    生成某一层级的待选检测框的anchors
    :param scales: 整形,表示某一层级中anchor在图片中的像素大小. 例如: [32, 64, 128]
    :param ratios: 一维数组,表示anchor的长宽比例. 例如: [0.5, 1, 2]
    :param shape: [height, width] 某一层级特征图的空间shape,由于有特征金字塔,
    所以存在多种不同层级的特征图.因而shape也会缩放
    :param feature_stride: 整形,表示某一层级中的特征图相对于图片以像素单位的步长.
    即表示特征图上的一步相当于原图上的feature_stride步. 配合上述shape即可获取图片尺寸
    :param anchor_stride: 整形,anchors某一层级在特征图上的像素步长. 比如说,
    为2时就是每个一个特征图的像素生成一次anchors. 但一般为1.
    :return anchors:
    """
    # 获取scales和ratios的笛卡尔集,即两两组合,并展开
    # scales不同层级的实际实现还需一层循环,
    # 由于层级与很多变量都有关联,故这里只考虑对于某一层级产生的anchor
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # 根据scales和ratios计算出出heights和widths
    # 即在一点中可以出现尺寸数*比例数的anchor
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # 计算出在特征图上的偏移量,
    # 即表示特征图中的每一像素点的左上角的对应于图片中像素点
    # 注意scale会影响feature_stride,
    # 每个层级都分别有不同的feature_stride与特征图的尺寸配适
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    # 横纵坐标两两组合
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # 枚举出shifts, widths, and heights的组合
    # 即x*y个点上有ratios种形状
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # 调整shape以获取(y,x)和(h,w)的列表
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # 转换为两个点的坐标组合形式(y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def display_anchors():
    """
    显示待选检测框anchors
    """
    assert config.BACKBONE in ["resnet50", "resnet101"]

    # 计算特征金字塔网络FPN即骨架网络的不同层级的shape,即是原尺寸/步长
    backbone_shapes = np.array(
        [[int(math.ceil(config.IMAGE_SHAPE[0] / stride)),
          int(math.ceil(config.IMAGE_SHAPE[0] / stride))]
         for stride in config.BACKBONE_STRIDES])

    # 生成待选检测框
    # 形式为[anchor_count, (y1, x1, y2, x2)]
    anchors = []
    scales = config.RPN_ANCHOR_SCALES
    # 对于特征金字塔网络即FPN的每一层级的anchors最好分开讨论
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], config.RPN_ANCHOR_RATIOS, backbone_shapes[i],
                                        config.BACKBONE_STRIDES[i], config.RPN_ANCHOR_STRIDE))
    anchors = np.concatenate(anchors, axis=0)
    # 打印生成待选检测框的信息
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

def dispaly_data_generator(dataset):
    """
    :param dataset:数据集,训练集或者验证集
    :return: data_generator
    """
    # Create data generator
    random_rois = 2000
    g = modellib.data_generator(
        dataset, config, shuffle=True, random_rois=random_rois,
        batch_size=1,
        detection_targets=True)
    return g


def train(end_epoch,dataset_name = "crack500", init_with = "没有,不想排除"):
    """
    训练模型
    :param end_epoch: 结束训练的epoch
    :return:
    """
    # 准备数据集
    if dataset_name == "crack500":
        dataset_train = Crack500Dataset()
        dataset_train.load_balloon("F:\\360downloads\\CRACK500\\", "train.txt")
        dataset_train.load_mask(0)
        dataset_train.prepare()
        dataset_val = Crack500Dataset()
        dataset_val.load_balloon("F:\\360downloads\\CRACK500\\", "val.txt")
        dataset_val.load_mask(0)
        dataset_val.prepare()
    else:
        dataset_train, dataset_val = prepare_datasets()
    # 以训练模式创建模型
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # TODO 选择一个权重初始化训练
    # init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # 加载在MS COCO上训练过的权值,但是我们的类别只有背景和
        # 混凝土裂缝两种,所以要跳过那些由于类别数目不同而导致的
        # 不同的网络层
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask","mrcnn_class"])
    elif init_with == "last":
        # 加载上次训练的权值并继续训练
        print("Loading weights from ", model.find_last())
        model.load_weights(model.find_last(), by_name=True)
    # 训练分为两个阶段:
    # 1. 只训练头部. 冻结所有的骨架网路层只训练初始化的网络层
    # (比如那些我们没有使用来自MS COCO的与训练的权重),
    # 在train()中传入参数'layers = 'heads''即可
    # 2. 微调所有网络层.对于一些简单的例子这甚至是多余的.
    # 因为MS COCO预训练的权重足够了. 而且全部训练的话会
    # 遇到显存不足的问题. 但是至少可以满足训练'layers = 5+'
    # model.train(dataset_train, dataset_val,
    #         learning_rate=config.LEARNING_RATE / 10,
    #         epochs=end_epoch,
    #         layers='heads')
    # 传递参数 layers="all" 以训练所有网络层.
    # 也可以传递正则表达式来匹配名字选择特定的网络层进行训练
    # 名字可以通过model.keras_model.trainable_weights查看
    # 值得一提的是 epochs表示训练到哪一epoch,
    # 即必须比上次的数值大才能开始训练
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE ,
                epochs=end_epoch,
                layers="5+")


def det(dataset_name = "crack500"):
    """
    随机检测和性能评估
    :return:
    """
    # 准备数据集
    # 准备数据集
    if dataset_name == "crack500":
        # dataset_train = Crack500Dataset()
        # dataset_train.load_balloon("F:\\360downloads\\CRACK500\\", "train.txt")
        # dataset_train.load_mask(0)
        # dataset_train.prepare()
        dataset_val = Crack500Dataset()
        dataset_val.load_balloon("F:\\360downloads\\CRACK500\\", "val.txt")
        dataset_val.load_mask(0)
        dataset_val.prepare()
    else:
        dataset_train, dataset_val = prepare_datasets()
    # # 创建推理配置
    #
    inference_config = InferenceConfig()
    #
    # # 以推理模式恢复模型
    # model = modellib.MaskRCNN(mode="inference",
    #                           config=inference_config,
    #                           model_dir=MODEL_DIR)
    # # 获取保存的权重的路径
    # # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    # # model_path = model.find_last()
    #
    # # 加载训练好的权值
    # print("Loading weights from ", model_path)
    # model.load_weights(model_path, by_name=True)


    # 在随机的图片上进行测试
    # TODO
    dataset = dataset_val
    image_id = random.choice(dataset.image_ids)
    # image_id = 14
    print(image_id)
    # 加载图片和元数据
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    #可视化原图
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names, figsize=(8, 8))
    #进行推理
    results = model.detect([original_image], verbose=1)
    #输入只有一张图片,
    r = results[0]
    #可视化原图+检测框+mask
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'])

    # TODO

    # Otsu_path = dataset.image_info[image_id]["path"]
    # Otsu_img = cv2.imread(Otsu_path, 0)
    # # threshold
    # height , width = Otsu_img.shape[:2]
    # # 高斯滤波后再采用Otsu阈值
    # blur = cv2.GaussianBlur(Otsu_img,
    #                         (height // Config.GaussianBlurFactor * 2 + 1, width // Config.GaussianBlurFactor * 2 + 1),
    #                         0)
    # blur_cons5 = cv2.GaussianBlur(Otsu_img, (5, 5), 0)
    # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret2, thresh_cons5 = cv2.threshold(blur_cons5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.figure('adaptive threshold', figsize=(12, 12))
    # plt.subplot(331), plt.imshow(Otsu_img, cmap='gray'), plt.title('original')
    # plt.subplot(337), plt.imshow(blur, cmap='gray'), plt.title('blur')
    # plt.subplot(338), plt.imshow(blur_cons5, cmap='gray'), plt.title('blur_cons5')
    # plt.subplot(339), plt.imshow(thresh_cons5, cmap='gray'), plt.title('thresh_cons5')
    # plt.subplot(334), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
    #
    # # blur = cv2.GaussianBlur(Otsu_img, (height//16 *2 +1, width//16 *2 +1), 0)
    # # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # plt.figure('adaptive threshold', figsize=(12, 12))
    # # plt.subplot(331), plt.imshow(Otsu_img, cmap='gray'), plt.title('original')
    # # plt.subplot(337), plt.imshow(blur, cmap='gray'), plt.title('blur')
    # #
    # # plt.subplot(334), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
    # # plt.show()
    #
    # # plt.imshow(r["original_masks"][0][0, :, :, 0])
    # # plt.imshow(r["original_masks"][0][1, :, :, 0])
    #
    # masks = r["original_masks"][0][0, :, :, 1]  # (N,x,y,IDs)[scores]
    # threshold = 0.5
    # y1, x1, y2, x2 = gt_bbox[0]
    # mask = utils.resize(masks, (y2 - y1, x2 - x1))
    # # mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    #
    # # Put the mask in the right location.
    # full_mask = np.zeros(original_image.shape[:2], dtype=np.float)
    # full_mask[y1:y2, x1:x2] = mask
    # # plt.imshow(full_mask, cmap='gray')
    # plt.subplot(333), plt.imshow(masks, cmap='gray'), plt.title('original_masks')
    # plt.subplot(332), plt.imshow(full_mask, cmap='gray'), plt.title('full_mask')
    #
    # thresh2 = utils.resize(thresh,(128,128))
    # # bool_mask = np.zeros(original_image.shape[:2], dtype=np.bool)
    #
    # # bool_mask = np.where( (thresh2==1 and full_mask >= 0.5) or (thresh2==0 and full_mask >=0.9), 1, 0).astype(np.bool)
    # alpha = config.MixAlpha
    # mix_mask = full_mask*alpha + (255.0-thresh2)*(1-alpha)
    # bool_mask = np.where(mix_mask > 0.5*255 , 1 , 0 ).astype(bool)
    # plt.subplot(335), plt.imshow(mix_mask, cmap='gray'), plt.title('mix_mask')
    # plt.subplot(336), plt.imshow(bool_mask, cmap='gray'), plt.title('bool_mask')
    #
    # plt.show()
    N = r["rois"].shape[0]

    def mix_thresh(image_id,float_masks,image_meta , verbose = False):
        N = len(float_masks)
        final_mask = np.zeros([128, 128, N]).astype(bool)
        for i in range(N):
            float_mask = float_masks[i]
            min_v = float_mask.min()
            max_v = float_mask.max()
            float_mask = (float_mask - min_v) * 255.0 / (max_v - min_v)
            Otsu_path = dataset.image_info[image_id]["path"]
            Otsu_img = cv2.imread(Otsu_path, 0)
            # threshold
            height, width = Otsu_img.shape[:2]
            # 高斯滤波后再采用Otsu阈值
            blur = cv2.GaussianBlur(Otsu_img, (
            height // Config.GaussianBlurFactor * 2 + 1, width // Config.GaussianBlurFactor * 2 + 1), 0)
            blur_cons5 = cv2.GaussianBlur(Otsu_img, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret2, thresh_cons5 = cv2.threshold(blur_cons5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if verbose:
                plt.figure('mix threshold', figsize=(12, 16))
                plt.subplot(341), plt.imshow(Otsu_img, cmap='gray'), plt.title('original')
                plt.subplot(347), plt.imshow(blur, cmap='gray'), plt.title('blur')
                plt.subplot(348), plt.imshow(blur_cons5, cmap='gray'), plt.title('blur_cons5')
                plt.subplot(349), plt.imshow(thresh_cons5, cmap='gray'), plt.title('thresh_cons5')
                plt.subplot(344), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
                plt.subplot(342), plt.imshow(float_mask), plt.title("float_mask")
            # plt.imshow(r["original_masks"][0][0, :, :, 0])
            # plt.imshow(r["original_masks"][0][1, :, :, 0])

            masks = r["original_masks"][0][i, :, :, 1]  # (N,x,y,IDs)[scores]
            threshold = 0.5

            # mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
            # Put the mask in the right location.
            # 调整至检测框
            # Put the mask in the right location.
            y1, x1, y2, x2 = r['rois'][i]

            full_mask = np.zeros(original_image.shape[:2], dtype=float)
            mask = utils.resize(masks, [y2 - y1, x2 - x1])
            full_mask[y1:y2, x1:x2] = float_mask[:, :]
            # plt.imshow(full_mask, cmap='gray')
            if verbose:
                plt.subplot(343), plt.imshow(masks, cmap='gray'), plt.title('original_size_masks')
                plt.subplot(345), plt.imshow(full_mask, cmap='gray'), plt.title('full_mask')
            y1, x1, y2, x2 = window = image_meta[7:11].astype(int)
            # 灰度图缩放到窗口
            thresh2 = np.ones([128,128])*255.0
            thresh2[y1:y2, x1:x2] = utils.resize(thresh, [y2 - y1, x2 - x1])


            # bool_mask = np.zeros(original_image.shape[:2], dtype=np.bool)

            # bool_mask = np.where( (thresh2==1 and full_mask >= 0.5) or (thresh2==0 and full_mask >=0.9), 1, 0).astype(np.bool)
            alpha = Config.MixAlpha
            shift = Config.MixShift
            beta = Config.MixBeta
            mix_mask = full_mask * alpha + (255.0 - thresh2 - shift) *beta
            bool_mask = np.where(mix_mask > 0.5 * 255, 1, 0).astype(bool)
            if verbose:
                plt.subplot(3,4,10), plt.imshow(mix_mask, cmap='gray'), plt.title('mix_mask')
                plt.subplot(3,4,11), plt.imshow(bool_mask, cmap='gray'), plt.title('bool_mask')
                plt.show()
            final_mask[:, :, i] = bool_mask

            # y1, x1, y2, x2  = window = image_meta[7:11].astype(int)
            # final_mask[y1:y2, x1:x2, i] = utils.resize(bool_mask,[y2 - y1, x2 - x1])

        return final_mask

    final_mask = mix_thresh(image_id,r['float_masks'],image_meta)
    # Put the mask in the right location.

    # reshape_mask = bool_mask.reshape([128,128,1])

    #TODO integrate_mask
    is_integrate = True
    if is_integrate:
        integrate_mask = np.zeros([128,128,1]).astype(bool)
        for i in range(r['masks'].shape[-1]):
            integrate_mask[:,:,0] = integrate_mask[:,:,0] | r['masks'][:,:,i]
            # integrate_mask = np.where(r['masks'][:,:,i], True, integrate_mask[:,:,i])
        integrate_mask_overlap = utils.compute_overlaps_masks(integrate_mask,gt_mask)
        print(f'integrate_mask_overlap = {integrate_mask_overlap}')
        canv = np.zeros([128, 128, 3])
        canv[:, :, 0] = integrate_mask[:, :, 0]
        canv[:, :, 1] = gt_mask[:, :, 0]
        plt.imshow(canv)
        plt.title("integrate_mask_and_gt_mask_IOU")
        plt.show()
    else:
        print(f'no_integrate')

    visualize.display_instances(original_image, r['rois'], final_mask, r['class_ids'],
                                dataset.class_names, r['scores'])

    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        r["rois"], r['class_ids'], r['scores'], r['masks'],
        iou_threshold = config.iou_threshold)
    gt_match, pred_match, opt_overlaps = utils.compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        r["rois"], r['class_ids'], r['scores'], final_mask,
        iou_threshold = config.iou_threshold)
    print(f"image_id = {image_id},path={dataset.image_info[image_id]['path']},overlaps = {overlaps},opt_overlaps = {opt_overlaps}")
    # END

    # 性能评估

    # 计算 VOC-Style mAP @ IoU=0.5
    # Intersection over Union
    iou_threshold = config.iou_threshold
    TEST_NUM = 100
    image_ids = np.random.choice(dataset.image_ids, TEST_NUM)
    image_ids = dataset.image_ids
    APs = []
    Ps = []
    Rs = []
    # OAPs = []
    # OPs = []
    # ORs = []
    OLs = []
    OOLs = []
    IOOLs = []
    overlaps_detail =[]
    TPs = RMs = PMs = 0

    visual = False
    # visual = True
    for image_id in image_ids:
        # 加载图片和元数据
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # 运行对象检测
        results = model.detect([image], verbose=0)
        r = results[0]


        opt_mask = mix_thresh(image_id,r['float_masks'],image_meta)
        if visual:
            # 可视化原图
            visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                        dataset.class_names, figsize=(8, 8))
            # 可视化原图+检测框+mask
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'])
            visualize.display_instances(image, r['rois'], opt_mask, r['class_ids'],
                                        dataset.class_names, r['scores'])

        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset_val.class_names, r['scores'])
        # 计算AP
        is_integrate = True
        if is_integrate:
            integrate_mask = np.zeros([128, 128, 1]).astype(bool)
            for i in range(opt_mask.shape[-1]):
                integrate_mask[:, :, 0] = integrate_mask[:, :, 0] | opt_mask[:, :, i]
                # integrate_mask = np.where(r['masks'][:,:,i], True, integrate_mask[:,:,i])
            integrate_mask_overlap = utils.compute_overlaps_masks(integrate_mask, gt_mask)
            print(f'integrate_mask_overlap = {integrate_mask_overlap}')

            if visual:
                canv = np.zeros([128, 128, 3])
                canv[:, :, 0] = integrate_mask[:, :, 0]
                canv[:, :, 1] = gt_mask[:, :, 0]
                plt.imshow(canv)
                plt.title(f"integrate_mask_and_gt_mask_IOU,id={image_id}")
                plt.show()
        else:
            print(f'no_integrate')
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=iou_threshold)


        APs.append(AP)
        Ps.append(precisions[1])
        Rs.append(recalls[1])
        OLs.append(overlaps.max())
        IOOLs.append(integrate_mask_overlap.max())
        # TODO 计算opt_overlaps

        gt_match, pred_match, opt_overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r['class_ids'], r['scores'], opt_mask,
            iou_threshold=iou_threshold)
        # AP, precisions, recalls, opt_overlaps = \
        #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                      r["rois"], r["class_ids"], r["scores"], opt_mask,iou_threshold=iou_threshold)
        OOLs.append(opt_overlaps.max())
        # OAPs.append(AP)
        # OPs.append(precisions[1])
        # ORs.append(recalls[1])

        print(f'id:{image_id},path={dataset.image_info[image_id]["path"]}, AP:{AP}, precisions:{precisions}, recalls:{recalls}, overlaps:{overlaps}, opt_overlaps:{opt_overlaps}')
        maxi = opt_overlaps.argmax()
        TP = np.sum((opt_mask[:,:,maxi] & gt_mask[:,:,0]))

        PM = np.sum(gt_mask)
        RM = np.sum(opt_mask[:,:,maxi])
        TPs += TP
        PMs += PM
        RMs += RM
        print(f'current precision:{TP / PM},recall:{TP / RM}')

    print(f"meanRecall @ IoU={iou_threshold*100}: ", np.mean(Rs))
    print(f"meanPrecision @ IoU={iou_threshold*100}: ", np.mean(Ps))
    print(f"meanOverlaps @ IoU={iou_threshold*100}: ", np.mean(OLs))
    print(OLs)
    print(f"mAP @ IoU={iou_threshold*100}: ", np.mean(APs))
    print(f"opt_overlaps @ IoU={iou_threshold*100}: ", np.mean(OOLs),OOLs)
    print(f"integrate_mask_overlap @ IoU={iou_threshold*100}: ", np.mean(IOOLs),IOOLs)
    print(f'global precision:{TPs / PMs},recall:{TPs / RMs}')

    x = 1

def simple_det(dataset_name = "crack500"):
    """
    单纯随机检测和性能评估
    :return:
    """
    # 准备数据集
    # 准备数据集
    if dataset_name == "crack500":
        # dataset_train = Crack500Dataset()
        # dataset_train.load_balloon("F:\\360downloads\\CRACK500\\", "train.txt")
        # dataset_train.load_mask(0)
        # dataset_train.prepare()
        dataset_val = Crack500Dataset()
        dataset_val.load_balloon("F:\\360downloads\\CRACK500\\", "val.txt")
        dataset_val.load_mask(0)
        dataset_val.prepare()
    else:
        dataset_train, dataset_val = prepare_datasets()
    # # 创建推理配置
    #
    inference_config = InferenceConfig()
    #
    # # 以推理模式恢复模型
    # model = modellib.MaskRCNN(mode="inference",
    #                           config=inference_config,
    #                           model_dir=MODEL_DIR)
    # # 获取保存的权重的路径
    # # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    # # model_path = model.find_last()
    #
    # # 加载训练好的权值
    # print("Loading weights from ", model_path)
    # model.load_weights(model_path, by_name=True)


    # 在随机的图片上进行测试
    dataset = dataset_val
    image_id = random.choice(dataset.image_ids)
    # image_id = 14
    print(image_id)
    # 加载图片和元数据
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    #可视化原图
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names, figsize=(8, 8))
    #进行推理
    results = model.detect([original_image], verbose=1)
    #输入只有一张图片,
    r = results[0]
    #可视化原图+检测框+mask
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'])

    # TODO

    # 性能评估

    # 计算 VOC-Style mAP @ IoU=0.5
    # Intersection over Union
    iou_threshold = config.iou_threshold
    image_ids = np.random.choice(dataset.image_ids, 100)
    APs = []
    Ps = []
    Rs = []
    OLs = []
    TPs = RMs = PMs = 0

    OOLs = []
    for image_id in image_ids:
        # 加载图片和元数据
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # 运行对象检测
        results = model.detect([image], verbose=0)
        r = results[0]
        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset_val.class_names, r['scores'])
        # 计算AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=iou_threshold)

        APs.append(AP)
        Ps.append(precisions[1])
        Rs.append(recalls[1])
        if len(overlaps) > 0:
            OLs.append(overlaps)

        # TODO 计算opt_overlaps
        # masks = r["original_masks"][0][0, :, :, 1]  # (N,x,y,IDs)[scores]
        # y1, x1, y2, x2 = gt_bbox[0]
        # mask = utils.resize(masks, (y2 - y1, x2 - x1))
        # full_mask = np.zeros(original_image.shape[:2], dtype=float)
        # full_mask[y1:y2, x1:x2] = mask
        # thresh2 = utils.resize(thresh, (128, 128))
        # mix_mask = full_mask * 0.55 + (255.0 - thresh2) * 0.45
        # bool_mask = np.where(mix_mask > 0.5 * 255, 1, 0).astype(bool)
        # reshape_mask = bool_mask.reshape([128, 128, 1])
        #
        # gt_match, pred_match, opt_overlaps = utils.compute_matches(
        #     gt_bbox, gt_class_id, gt_mask,
        #     r["rois"], r['class_ids'], r['scores'], reshape_mask,
        #     iou_threshold=0.5)
        # OOLs.append(opt_overlaps)
        # end
        maxi = overlaps.argmax()
        TP = np.sum((r['masks'][:, :, maxi] & gt_mask[:, :, 0]))

        PM = np.sum(gt_mask)
        RM = np.sum(r['masks'][:, :, maxi])
        TPs += TP
        PMs += PM
        RMs += RM
        print(f'current precision:{TP / PM},recall:{TP / RM}')

        print(f'id:{image_id}, AP:{AP}, precisions:{precisions}, recalls:{recalls}, overlaps:{overlaps}')

    print(f"meanRecall @ IoU={iou_threshold*100}: ", np.mean(Rs))
    print(f"meanPrecision @ IoU={iou_threshold*100}: ", np.mean(Ps))
    print(f"meanOverlaps @ IoU={iou_threshold*100}: ", np.mean([a.max() for a in OLs]))
    print(f"mAP @ IoU={iou_threshold*100}: ", np.mean(APs))
    # print(f"opt_overlaps @ IoU={iou_threshold*100}: ", np.mean(OOLs))
    print(f'global precision:{TPs / PMs},recall:{TPs / RMs}')

def det_crack500(min2 = False):
    """
    随机检测和性能评估
    :return:
    """
    # 准备数据集
    dataset_train = Crack500Dataset()
    if min2:
        dataset_train = Crack500_min2_Dataset()
    dataset_train.load_balloon("F:\\360downloads\\CRACK500\\", "train.txt")
    dataset_train.load_mask(0)
    dataset_train.prepare()
    # 创建推理配置

    inference_config = InferenceConfig()

    # 以推理模式恢复模型
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    # 获取保存的权重的路径
    # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    # model_path = model.find_last()

    # 加载训练好的权值
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # 在随机的图片上进行测试
    dataset = dataset_train
    image_id = random.choice(dataset.image_ids)
    # image_id = 544
    image_id = 2701
    print(f'visual_image_id = {image_id}')
    # 加载图片和元数据
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    #可视化原图
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names, figsize=(8, 8))
    #进行推理
    results = model.detect([original_image], verbose=1)
    #输入只有一张图片,
    r = results[0]
    #可视化原图+检测框+mask
    float_masks = r['float_masks']
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'])

    # TODO

    Otsu_path = dataset.image_info[image_id]["path"]
    Otsu_img = cv2.imread(Otsu_path, 0)
    # threshold
    # 高斯滤波后再采用Otsu阈值
    blur = cv2.GaussianBlur(Otsu_img, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure('adaptive threshold', figsize=(12, 8))
    plt.subplot(231), plt.imshow(Otsu_img, cmap='gray'), plt.title('original')
    plt.subplot(234), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
    # plt.show()

    # plt.imshow(r["original_masks"][0][0, :, :, 0])
    # plt.imshow(r["original_masks"][0][1, :, :, 0])

    masks = r["original_masks"][0][0, :, :, 1]  # (N,x,y,IDs)[scores]
    threshold = 0.5

    # mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(original_image.shape[:2], dtype=float)
    # TODO  Translate normalized coordinates in the resized image to pixel
    def comp_small_wind():
        # coordinates in the original image before resizing
        # window:real image is excluding the padding.
        window = [0, 0, original_image.shape[0], original_image.shape[1]]
        window = image_meta[7:11]
        window = utils.norm_boxes(window, original_image.shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = r["detections"][0,0,:4]
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, image_meta[1:3])
        return boxes
    def comp_mid_wind():
        window = image_meta[7:11]
        return window.astype(int)
    # y1, x1, y2, x2 = comp_wind()
    if len(gt_bbox) >= 1:
        # 没有裂缝就完了
        y1, x1, y2, x2 = gt_bbox[0]
        mask = utils.resize(masks, (y2 - y1, x2 - x1))
        full_mask[y1:y2, x1:x2] = mask
        # plt.imshow(full_mask, cmap='gray')
        plt.subplot(233), plt.imshow(masks, cmap='gray'), plt.title('original_masks')
        plt.subplot(232), plt.imshow(full_mask, cmap='gray'), plt.title('full_mask')

        thresh2 = utils.resize(thresh,(128,128))
        # bool_mask = np.zeros(original_image.shape[:2], dtype=np.bool)

        # bool_mask = np.where( (thresh2==1 and full_mask >= 0.5) or (thresh2==0 and full_mask >=0.9), 1, 0).astype(np.bool)
        mix_mask = full_mask*0.55 + (255.0-thresh2)*0.45
        bool_mask = np.where(mix_mask > 0.5*255 , 1 , 0 ).astype(bool)
        plt.subplot(235), plt.imshow(mix_mask, cmap='gray'), plt.title('mix_mask')
        plt.subplot(236), plt.imshow(bool_mask, cmap='gray'), plt.title('bool_mask')

        plt.show()
        # Put the mask in the right location.
        reshape_mask = utils.resize(bool_mask,[128,128,1])
        visualize.display_instances(original_image, r['rois'], reshape_mask, r['class_ids'],
                                    dataset.class_names, r['scores'])
        window_mask = np.zeros([128,128,1], dtype=float)
        y1, x1, y2, x2 = comp_mid_wind()
        window_mask[y1:y2, x1:x2,:] = utils.resize(reshape_mask,[y2-y1, x2-x1,1])
        visualize.display_instances(original_image, r['rois'], window_mask, r['class_ids'],
                                    dataset.class_names, r['scores'])

        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r['class_ids'], r['scores'], r['masks'],
            iou_threshold =config.iou_threshold)
        gt_match, pred_match, opt_overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r['class_ids'], r['scores'], reshape_mask,
            iou_threshold=config.iou_threshold)
        print("image_id = ",image_id,"overlaps = ", overlaps, "opt_overlaps = ",opt_overlaps)

        # END

    # 性能评估

    # 计算 VOC-Style mAP @ IoU=0.5
    # Intersection over Union
    iou_threshold = config.iou_threshold
    image_ids = np.random.choice(dataset.image_ids, 100)
    APs = []
    Ps = []
    Rs = []
    OLs = []
    OOLs = []
    for image_id in image_ids:
        # 加载图片和元数据
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # 运行对象检测
        results = model.detect([image], verbose=0)
        r = results[0]



        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset_val.class_names, r['scores'])
        # 计算AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=iou_threshold)




        # TODO 计算opt_overlaps
        if len(gt_bbox) < 1 :
            continue
        y1, x1, y2, x2 = gt_bbox[0]
        masks = r["original_masks"][0][0, :, :, 1]  # (N,x,y,IDs)[scores]
        mask = utils.resize(masks, (y2 - y1, x2 - x1))
        full_mask = np.zeros(original_image.shape[:2], dtype=float)
        full_mask[y1:y2, x1:x2] = mask
        thresh2 = utils.resize(thresh, (128, 128))
        mix_mask = full_mask * 0.6 + (255.0 - thresh2) * 0.4
        bool_mask = np.where(mix_mask > 0.5 * 255, 1, 0).astype(bool)
        reshape_mask = bool_mask.reshape([128, 128, 1])

        gt_match, pred_match, opt_overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r["rois"], r['class_ids'], r['scores'], reshape_mask,
            iou_threshold=config.iou_threshold)
        if len(overlaps) < 1 or len(gt_match) < 1 or len(opt_overlaps) < 1:
            continue
        OOLs.append(opt_overlaps)
        APs.append(AP)
        Ps.append(precisions[1])
        Rs.append(recalls[1])
        OLs.append(overlaps)

        print(f'id:{image_id},AP:{AP}, precisions:{precisions}, recalls:{recalls}, overlaps:{overlaps}, opt_overlaps:{opt_overlaps}')


    print(f"meanRecall @ IoU={iou_threshold*100}: ", np.mean(Rs))
    print(f"meanPrecision @ IoU={iou_threshold*100}: ", np.mean(Ps))
    print(f"meanOverlaps @ IoU={iou_threshold*100}: ", np.mean(OLs))
    print(f"mAP @ IoU={iou_threshold*100}: ", np.mean(APs))
    print(f"opt_overlaps @ IoU={iou_threshold*100}: ", np.mean(OOLs))

def load_infer_model(init_with_last = False):
    inference_config = InferenceConfig()
    # 以推理模式恢复模型
    global model
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    # 获取保存的权重的路径
    # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    model_path = os.path.join(ROOT_DIR, "logs/cracks20220714T1112/mask_rcnn_cracks_0140.h5")
    # 用mini 最后一层
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220814T1729/mask_rcnn_cracks_0001.h5")

    # model_path = model.find_last()
    if init_with_last:
        model_path = model.find_last()
    # 加载训练好的权值
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
def det_single(base_path = None):
    """
    随机检测和性能评估
    :return:
    """
    # 准备数据集

    # 创建推理配置
    """
    inference_config = InferenceConfig()

    # 以推理模式恢复模型
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    # 获取保存的权重的路径
    # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    # model_path = model.find_last()

    # 加载训练好的权值
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    """

    # 加载图片和元数据
    # base_path = "F:\\360downloads\\CRACK500\\traincrop\\20160222_081031_1_721"
    # base_path = "F:\\360downloads\\CRACK500\\traincrop\\20160307_145107_641_721"

    if base_path == None:
        base_path = "F:\\new_work\\concrete_crack\\test\\test1"
    img_path = base_path+".jpg"
    mask_path = base_path+".png"
    image = skimage.io.imread(img_path)
    original_mask = skimage.io.imread(mask_path)

    # original_image = np.resize(image,[128,128,3])
    DIM = config.IMAGE_MAX_DIM
    molded_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=DIM,
        min_scale=0,
        max_dim=DIM,
        mode="square")
    cvresize_img = cv2.resize(image,[DIM,DIM])
    original_image = cvresize_img

    #可视化原图

    #进行推理
    results = model.detect([cvresize_img], verbose=1)
    #输入只有一张图片,
    r = results[0]
    #可视化原图+检测框+mask
    float_masks = r['float_masks']
    if len(float_masks) == 0:
        print("None detection")
        return None

    plt.figure("original_image",figsize=(12,8))
    plt.subplot(221),    plt.imshow(molded_image),    plt.title("original_image")
    plt.subplot(222),    plt.imshow(cvresize_img),    plt.title("cvresize_img")
    plt.subplot(223),    plt.imshow(original_mask),    plt.title("original_mask")
    plt.show()
    boxes = np.zeros([1,4])
    boxes[0,:] = [0,0,DIM,DIM]
    class_ids = np.array([1])
    scores = np.array([1.0])
    mode_mask = utils.resize(original_mask, (DIM, DIM,1))
    visualize.display_instances(original_image, boxes, mode_mask, class_ids,
                                ["bg", "crack"], scores)

    single_mask_temp = np.reshape(r["masks"][:,:,0],[DIM,DIM,1])
    visualize.display_instances(original_image, r["rois"], r["masks"], r["class_ids"],
                                ["bg", "crack"], r["scores"])
    # TODO
    overlaps1 = utils.compute_overlaps_masks(mode_mask,r["masks"])
    print(f'overlaps_mode_mask_ = {overlaps1}')
    N = r["rois"].shape[0]

    def mix_thresh( float_masks, image_meta, verbose=True):
        N = len(float_masks)
        final_mask = np.zeros([DIM, DIM, N]).astype(bool)
        for i in range(N):
            float_mask = float_masks[i]
            # min_v = float_mask.min()
            # max_v = float_mask.max()
            # float_mask = (float_mask - min_v) * 255.0 / (max_v - min_v)
            Otsu_path = img_path
            Otsu_img = cv2.imread(Otsu_path, 0)
            # threshold
            height, width = Otsu_img.shape[:2]
            # 高斯滤波后再采用Otsu阈值
            core_size = np.min([height,width])// Config.GaussianBlurFactor * 2 + 1
            blur = cv2.GaussianBlur(Otsu_img, ( core_size, core_size), 0)
            blur_cons5 = cv2.GaussianBlur(Otsu_img, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret2, thresh_cons5 = cv2.threshold(blur_cons5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if verbose:
                plt.figure('mix threshold', figsize=(12, 16))
                plt.subplot(341), plt.imshow(Otsu_img, cmap='gray'), plt.title('original')
                plt.subplot(347), plt.imshow(blur, cmap='gray'), plt.title('blur')
                plt.subplot(348), plt.imshow(blur_cons5, cmap='gray'), plt.title('blur_cons5')
                plt.subplot(349), plt.imshow(thresh_cons5, cmap='gray'), plt.title('thresh_cons5')
                plt.subplot(344), plt.imshow(thresh, cmap='gray'), plt.title('otsu')
                plt.subplot(342), plt.imshow(float_mask), plt.title("float_mask")
            # plt.imshow(r["original_masks"][0][0, :, :, 0])
            # plt.imshow(r["original_masks"][0][1, :, :, 0])

            masks = r["original_masks"][0][i, :, :, 1]  # (N,x,y,IDs)[scores]
            threshold = 0.5

            # mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
            # Put the mask in the right location.
            # 调整至检测框
            # Put the mask in the right location.
            y1, x1, y2, x2 = r['rois'][i]

            full_mask = np.zeros(original_image.shape[:2], dtype=float)
            mask = utils.resize(masks, [y2 - y1, x2 - x1])
            full_mask[y1:y2, x1:x2] = float_mask[:, :]
            # plt.imshow(full_mask, cmap='gray')
            if verbose:
                plt.subplot(343), plt.imshow(masks, cmap='gray'), plt.title('original_size_masks')
                plt.subplot(345), plt.imshow(full_mask, cmap='gray'), plt.title('full_mask')
            # TODO
            y1, x1, y2, x2 = 0,0,DIM,DIM #window = image_meta[7:11].astype(int)
            # 灰度图缩放到窗口
            thresh2 = np.ones([DIM, DIM]) * 255.0
            thresh2[y1:y2, x1:x2] = utils.resize(thresh, [y2 - y1, x2 - x1])

            # bool_mask = np.zeros(original_image.shape[:2], dtype=np.bool)

            # bool_mask = np.where( (thresh2==1 and full_mask >= 0.5) or (thresh2==0 and full_mask >=0.9), 1, 0).astype(np.bool)
            alpha = Config.MixAlpha
            shift = Config.MixShift
            beta = Config.MixBeta
            mix_mask = full_mask * alpha + (255.0 - thresh2 - shift) * beta
            bool_mask = np.where(mix_mask > 0.5 * 255, 1, 0).astype(bool)
            if verbose:
                plt.subplot(3, 4, 10), plt.imshow(mix_mask, cmap='gray'), plt.title('mix_mask')
                plt.subplot(3, 4, 11), plt.imshow(bool_mask, cmap='gray'), plt.title('bool_mask')
                plt.show()
            final_mask[:, :, i] = bool_mask

            # y1, x1, y2, x2  = window = image_meta[7:11].astype(int)
            # final_mask[y1:y2, x1:x2, i] = utils.resize(bool_mask,[y2 - y1, x2 - x1])

        return final_mask

    final_mask = mix_thresh(float_masks,[0,0,DIM,DIM])

    # reshape_mask = utils.resize(bool_mask,[128,128,1])
    # visualize.display_instances(original_image, r['rois'], reshape_mask, r['class_ids'],
    #                             ["bg","crack"], r['scores'])
    # window_mask = np.zeros([128,128,1], dtype=float)
    # window_mask[y1:y2, x1:x2,:] = utils.resize2(mix_mask,[y2-y1, x2-x1,1])
    # window_bool_mask = np.where(window_mask > 0.5*255 , 1 , 0 ).astype(bool)
    # final_mask = np.reshape(bool_mask, [128, 128, 1])
    single_box_temp = np.reshape(r['rois'][0],[1,4])
    visualize.display_instances(original_image,r['rois'] , final_mask, r['class_ids'],
                                ["bg","crack"], r['scores'])
    overlaps2 = utils.compute_overlaps_masks(mode_mask, final_mask)
    print(f'overlaps2 = {overlaps2}')

    return final_mask

    # END

def check_crack500():
    dataset_train = Crack500Dataset()
    dataset_train.load_balloon("F:\\360downloads\\CRACK500\\", "train.txt")
    dataset_train.load_mask(0)
    dataset_train.prepare()
    dataset_val = Crack500Dataset()
    dataset_val.load_balloon("F:\\360downloads\\CRACK500\\", "val.txt")
    dataset_val.load_mask(0)
    dataset_val.prepare()
    display_num = 4
    image_ids = np.random.choice(dataset_train.image_ids, display_num)
    for image_id in image_ids:
        # 从文件系统中读取图片
        image = dataset_train.load_image(image_id)
        # 获取mask和类别标识符
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

def print_hi(name):
    print(f'Hi, {name}')
    print(tf.__version__)
    print(ROOT_DIR)

def split_det(name):
    original_image = Image.open(f"F:\\new_work\\concrete_crack\\test\\{name}.jpg")
    iw, ih = original_image.size
    original_mask = Image.open(f"F:\\new_work\\concrete_crack\\test\\{name}.png")

    split_level = 2
    step = split_level*2 -1
    patch_w = iw /split_level
    patch_h = ih /split_level
    step_long_w = patch_w /2
    step_long_h = patch_h /2
    for jw in range(step):
        for jh in range(step):
            region = original_image.crop([jw * step_long_w, jh * step_long_h,
                                jw * step_long_w + patch_w, jh * step_long_h + patch_h] )
            region.save(f'./test/{name}_{jw}{jh}.jpg')
            region = original_mask.crop([jw * step_long_w, jh * step_long_h,
                                         jw * step_long_w + patch_w, jh * step_long_h + patch_h])
            region.save(f'./test/{name}_{jw}{jh}.png')
    split_detection(name)

def split_detection(name):
    split_level = 2
    step = split_level*2 -1
    patch_w = 128
    patch_h = 128
    step_long_w = patch_w // 2
    step_long_h = patch_h // 2

    integrate_mask = np.zeros([128*split_level,128*split_level])
    for jw in range(step):
        for jh in range(step):
            final_mask = det_single(base_path=f"F:\\new_work\\concrete_crack\\test\\{name}_{jw}{jh}")
            if type(final_mask) == type(None):
                continue
            # path = r"./001.jpg"  # 图片路径
            # img = Image.open(path)  # 打开图片
            # img.save("1.jpg")  # 将图片保存为1.jpg
            tp = integrate_mask[jh * step_long_h: jh * step_long_h + patch_h,
                 jw * step_long_w: jw * step_long_w + patch_w]

            tmp = np.ones([128, 128]) * 255
            t = integrate_mask[jh * step_long_h: jh * step_long_h + patch_h,
                jw * step_long_w: jw * step_long_w + patch_w] * 0.5 + tmp * 0.5

            integrate_mask[jh * step_long_h : jh * step_long_h + patch_h,
                jw * step_long_w : jw * step_long_w + patch_w] = \
            np.where(final_mask[:,:,0] == True,
                     integrate_mask[jh * step_long_h : jh * step_long_h + patch_h,
                     jw * step_long_w : jw * step_long_w + patch_w] * 0.5 + tmp * 0.5,
                     integrate_mask[jh * step_long_h:jh * step_long_h + patch_h,
                     jw * step_long_w: jw * step_long_w + patch_w]
                     )

    # img = Image.fromarray(integrate_mask)
    # img.convert("L").save("./test/integrate_mask.jpg")
    # cv2.imwrite('test.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    cv2.imwrite(f'./test/integrate_mask_{name}.jpg', integrate_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

def split_eval(name):
    final_mask = cv2.imread(f'./test/integrate_mask_{name}.jpg',cv2.IMREAD_GRAYSCALE)
    mode_mask = cv2.imread(f'./test/{name}.png',cv2.IMREAD_GRAYSCALE)
    w, h = mode_mask.shape[:2]
    final_mask = utils.resize(final_mask,[w,h])
    final_mask = np.reshape(final_mask,[w,h,1])
    mode_mask = np.reshape(mode_mask,[w,h,1])
    final_mask = np.where(final_mask > 25 , True, False)
    mode_mask = np.where(mode_mask > 1, True, False)

    overlaps = utils.compute_overlaps_masks(mode_mask, final_mask)
    print(f'split_integrate_overlaps{overlaps}')
    canv= np.zeros([w,h,3])
    canv[:,:,0] = final_mask[:,:,0]
    canv[:,:,1] = mode_mask[:,:,0]

    plt.imshow(final_mask), plt.title("final_mask")
    plt.show()

    plt.imshow(mode_mask), plt.title("mode_mask")
    plt.show()
    plt.imshow(canv)
    plt.show()

    huge_mask = det_single(f'./test/{name}')
    if type(huge_mask) == type(None):
        return
    huge_mask2 = utils.resize(huge_mask,[w,h])
    plt.imshow(huge_mask), plt.title("huge_mask")
    plt.show()
    final_integrate_mask = final_mask
    final_integrate_mask = np.where(final_integrate_mask == True, True, huge_mask2 == True)
    overlaps2 = utils.compute_overlaps_masks(mode_mask, final_integrate_mask)
    print(f'one_overlaps{overlaps2}')
    plt.imshow(final_integrate_mask), plt.title("final_integrate_mask")
    plt.show()
    x =1

if __name__ == '__main__':
    # print_hi('PyCharm')
    # load_infer_model(init_with_last = True)
    load_infer_model()
    # check_dataset()
    # display_anchors()
    # train(140,init_with="last")
    # train(2,init_with="last")
    # train(1)
    det()
    # simple_det()
    # config.display()
    # det_crack500(min2 = True)
    # det_single('./test/test6')
    # det_single('./test/test5')
    # split_det("test5")
    # split_eval("test5")
    # split_detection()


