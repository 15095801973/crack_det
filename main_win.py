import math

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
import tensorflow as tf

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
import mrcnn.model_win as modellib
from mrcnn.model_win import log
MODEL_DIR = os.path.join(ROOT_DIR,"logs")
DATASETS_DIR = os.path.join(ROOT_DIR,"datasets/crack_huge")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

#device_count={'cpu':0}
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
    TRAIN_ROIS_PER_IMAGE = 200

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
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]


    # 因为数据集是混凝土表面裂缝,往往贯穿整个图片,所以必须
    # 至少设置一个128的anchors才能捕获
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    USE_MINI_MASK = False
    #False
    ROI_POSITIVE_RATIO = 0.33 #0.33

    # BACKBONE_STRIDES = [8, 16, 32, 64, 128]
    # POOL_SIZE = 14
    MASK_POOL_SIZE = 28
    POOL_SIZE = 7 # 7
    MASK_SHAPE = [56, 56]
    POST_NMS_ROIS_TRAINING = 32
    POST_NMS_ROIS_INFERENCE = 32
    DETECTION_MAX_INSTANCES = 32
    # 每张图片的训练感兴趣区域,不需要太大,训练集里一张图片只有一两条裂缝
    # 至少我标注得是这样,对于一些形状丰富的可能需要几个检测才能满足
    # 能够保证取到正感兴趣区域.
    TRAIN_ROIS_PER_IMAGE = 32
    FPN_CLASSIF_FC_LAYERS_SIZE = 32
    # 每个epoch训练多少次
    STEPS_PER_EPOCH = 1000
    # Weight decay regularization l2
    WEIGHT_DECAY = 0.001 #0.0001

    LEARNING_RATE = 0.001 # 0.001
    LEARNING_MOMENTUM = 0.9 #0.9

    # 每个epoch验证多少次
    VALIDATION_STEPS = 2

    LOSS_WEIGHTS = {
        # "rpn_class_loss": 0.,
        # "rpn_bbox_loss": 0.,
        # "mrcnn_class_loss": 0.,
        # "mrcnn_bbox_loss": 0.0,
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.
    }

config = CracksConfig2()

# class InferenceConfig(CracksConfig):
class InferenceConfig(CracksConfig2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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
            # print("height, width = ", height, width)
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


def train(end_epoch,dataset_name = "crack500", init_with = "coco"):
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
        #dataset_train, dataset_val = prepare_datasets()
        print("error")
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
        # 现在fpn也不行了
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","fpn",
                                    "mrcnn_bbox", "mrcnn_mask","mrcnn_class","mrcnn",
                                    "mrcnn_mask_conv1", "mrcnn_mask_conv2","mrcnn_mask_conv3", "mrcnn_mask_conv4",
                                    "mrcnn_mask_bn1", "mrcnn_mask_bn2", "mrcnn_mask_bn3","mrcnn_mask_bn4",
                                    "mrcnn_mask_deconv",
                                    "mrcnn_class_bn1", "mrcnn_class_bn2", "mrcnn_class_conv1", "mrcnn_class_conv2"
                                    ])
    elif init_with == "last":
        # 加载上次训练的权值并继续训练
        print("Loading weights from ", model.find_last())
        model.load_weights(model.find_last(), by_name=True)
        # model.load_weights("logs/cracks20220818T0800/mask_rcnn_cracks_0001.h5", by_name=True)
    elif init_with == "crack140":
        # path = "logs/cracks20220714T1112/mask_rcnn_cracks_0140.h5"
        # path = "logs/cracks20220821T1746/mask_rcnn_cracks_0021.h5"
        path = "logs/cracks20220823T0905/mask_rcnn_cracks_0010.h5"
        print("Loading weights from ", path)
        # 自动跳过不符合的变量
        model.load_weights(path, by_name=True,
                           exclude=[
                           #     # "mrcnn_class_logits",
                           #          "mrcnn_bbox_fc",
                           #          # "fpn",
                           #          "mrcnn_bbox", "mrcnn_mask","mrcnn_class",
                                        "mrcnn_mask",
                                    "mrcnn_mask_conv1", "mrcnn_mask_conv2","mrcnn_mask_conv3", "mrcnn_mask_conv4",
                                    "mrcnn_mask_bn1", "mrcnn_mask_bn2", "mrcnn_mask_bn3","mrcnn_mask_bn4",
                           #          "mrcnn_mask_deconv",
                           #          "mrcnn_class_bn1", "mrcnn_class_bn2", "mrcnn_class_conv1", "mrcnn_class_conv2"
                                    ]
                           )

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
    # layers = r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_mask)|(mrcnn\_mask\_.*)|(rpn\_.*)|(fpn\_.*)"
    # layers = r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fpn\_.*)"
    # layers = "heads"
    layers = r"(mrcnn\_mask)|(mrcnn\_mask\_.*)"
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=end_epoch,
                layers=layers)

    #r"(mrcnn\_.*)"

def load_infer_model(init_with = "last"):
    inference_config = InferenceConfig()
    # 以推理模式恢复模型
    global model
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    # 获取保存的权重的路径
    # TODO 可以设置为一个特定的权值的路径,也可以直接使用最后一次的权值
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220311T1933/mask_rcnn_shapes_0037.h5")
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220714T1112/mask_rcnn_cracks_0140.h5")
    # 用mini 最后一层
    model_path = os.path.join(ROOT_DIR, "logs/cracks20220814T1729/mask_rcnn_cracks_0001.h5")

    # model_path = model.find_last()
    if init_with == "last":
        model_path = model.find_last()
    elif init_with == "order":
        # model_path = "logs/cracks20220714T1112/mask_rcnn_cracks_0140.h5"
        model_path = "logs/cracks20220821T1746/mask_rcnn_cracks_0021.h5"
        # print("Loading weights from ", path)
        # 自动跳过不符合的变量
        # model.load_weights(path, by_name=True)
    # 加载训练好的权值
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    print("END Loading weights from ", model_path)


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

    mode_mask = utils.resize_mask(np.expand_dims(original_mask, axis = 2), scale, padding, crop)

    cvresize_img = cv2.resize(image,[DIM,DIM])
    original_image = cvresize_img

    #可视化原图

    #进行推理
    results = model.detect([molded_image], verbose=1)
    #输入只有一张图片,
    r = results[0]
    #可视化原图+检测框+mask
    # 1为实例
    float_masks = r['float_masks'][:, :, :, 1]
    if len(float_masks) == 0:
        print("None detection")
        return None

    plt.figure("original_image",figsize=(12,8))
    plt.subplot(321),    plt.imshow(molded_image),    plt.title("original_image")
    plt.subplot(322),    plt.imshow(cvresize_img),    plt.title("cvresize_img")
    plt.subplot(323),    plt.imshow(original_mask),    plt.title("original_mask")
    plt.subplot(325),    plt.imshow(mode_mask),    plt.title("mode_mask")
    plt.subplot(324),    plt.imshow(float_masks[0], cmap="gray"),    plt.title("pre_mask")
    plt.show()

    shift_masks = np.transpose(float_masks, [1, 2, 0])
    single_box_temp = np.reshape(r['rois'][0],[1,4])
    #rois_2没有经过极大值抑制等
    # molded_image  or  original_image
    visualize.display_instances(molded_image,r['rois'] * DIM , shift_masks, r['class_ids'].astype(int),
                                ["bg","crack"], r['scores'])
    bool_masks = np.where(shift_masks > 0.4, 1, 0)
    overlaps2 = utils.compute_overlaps_masks(mode_mask, bool_masks)
    print(f'overlaps2 = {overlaps2}')

    return float_masks

    # END

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
        # 训练集
        dataset_train = CrackDataset()
        dataset_train.load_balloon(DATASETS_DIR, "train")
        dataset_train.prepare()
        # 验证集
        dataset_val = CrackDataset()
        dataset_val.load_balloon(DATASETS_DIR, "val")
        dataset_val.prepare()
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
    DIM = config.IMAGE_MAX_DIM
    dataset = dataset_val
    # 性能评估
    # 计算 VOC-Style mAP @ IoU=0.5
    # Intersection over Union
    iou_threshold = config.iou_threshold
    image_ids = np.random.choice(dataset.image_ids, 10)
    #
    scan_all = True
    if scan_all:
        image_ids = dataset.image_ids
    APs = []
    Ps = []
    Rs = []
    OLs = []
    OOLs = []
    IOOLs = []
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
        float_masks = r['float_masks'][:, :, :, 1]
        shift_masks = np.transpose(float_masks, [1, 2, 0])



        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset_val.class_names, r['scores'])
        # 计算AP

        # AP, precisions, recalls, overlaps = \
        #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                      r['rois'] * DIM, r['class_ids'].astype(int), r["scores"], shift_masks ,iou_threshold=iou_threshold)
        # APs.append(AP)
        # Ps.append(precisions[1])
        # Rs.append(recalls[1])
        # TODO 计算opt_overlaps
        bool_mask = np.where(shift_masks > .4, 1, 0)

        overlaps = utils.compute_overlaps_masks(gt_mask, bool_mask)
        OLs.append(overlaps.max())

        maxi = overlaps.argmax()
        TP = np.sum((bool_mask[:, :, maxi] & gt_mask[:, :, 0]))

        PM = np.sum(gt_mask)
        RM = np.sum(bool_mask[:, :, maxi])
        TPs += TP
        PMs += PM
        RMs += RM
        Ps.append(TP / PM)
        Rs.append(TP / RM)

        if visual:
            # 交并比可视化
            canv = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3])
            canv[:, :, 0] = np.where(gt_mask[:, :, 0] > .5, 1., 0.)
            canv[:, :, 1] = np.where(bool_mask[:, :, maxi] > .5, 1., 0.)
            # 可视化原图
            visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                        dataset.class_names, figsize=(8, 8))
            plt.figure("original_image", figsize=(12, 8))
            plt.subplot(221), plt.imshow(image), plt.title("original_image")
            plt.subplot(222), plt.imshow(canv), plt.title(str(float(overlaps.max())))
            plt.subplot(223), plt.imshow(gt_mask), plt.title("original_mask")
            plt.subplot(224), plt.imshow(float_masks[0], cmap="gray"), plt.title("pre_mask")
            plt.show()
            # 可视化原图+检测框+mask
            visualize.display_instances(image, r['rois_2'] * DIM, bool_mask, r['class_ids'].astype(int),
                                        dataset.class_names, r['scores'], title=str(float(overlaps.max())))



        print(f'id:{image_id},path={dataset.image_info[image_id]["path"]}, overlaps:{overlaps}')

        print(f'current precision:{TP / PM},recall:{TP / RM}')

    # print(f"meanRecall @ IoU={iou_threshold*100}: ", np.mean(Rs))
    # print(f"meanPrecision @ IoU={iou_threshold*100}: ", np.mean(Ps))
    print(f"meanOverlaps @ IoU={iou_threshold*100}: ", np.mean(OLs))
    print(f"Overlaps @ IoU={iou_threshold*100}: ", OLs)

    # print(f"mAP @ IoU={iou_threshold*100}: ", np.mean(APs))
    # print(f"opt_overlaps @ IoU={iou_threshold*100}: ", np.mean(OOLs),OOLs)
    # print(f"integrate_mask_overlap @ IoU={iou_threshold*100}: ", np.mean(IOOLs),IOOLs)
    print(f'global precision:{TPs / PMs},recall:{TPs / RMs}')
    print(f'precision:{Ps},recall:{Rs}')

    x = 1
def one_test(init_with="last"):
    load_infer_model(init_with = init_with)
    # det_single('./test/test5')
    # det_single('./test/test6')
    det_single('./test/test7')
    det_single('./test/test9')
    det_single('./test/test1')
    det_single('./test/test2')
def two_test(init_with = "last"):
    load_infer_model(init_with = init_with)
    det()
# with tf.device('/cpu:0'):
    #     v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
    #     v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
    #     sumV12 = v1 + v2
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #     print(sess.run(sumV12))
if __name__ == '__main__':

    # print_hi('PyCharm')
    # load_infer_model(init_with_last = True)
    # load_infer_model()
    # check_dataset()
    # display_anchors()
    # train(140,init_with="last")
    # train(24, init_with="last")
    train(32, init_with="crack140")
    # train(5, init_with="coco")
    # det()
    # simple_det()
    # config.display()
    # det_crack500(min2 = True)
    # det_single('./test/test6')
    # two_test("last")
    # one_test("last")
    # split_det("test5")
    # split_eval("test5")
    # split_detection()
    print("\nend")
