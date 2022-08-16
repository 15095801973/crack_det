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
import mrcnn.model_win as modellib
from mrcnn.model_win import log
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
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100
    # 每张图片的训练感兴趣区域,不需要太大,训练集里一张图片只有一两条裂缝
    # 至少我标注得是这样,对于一些形状丰富的可能需要几个检测才能满足
    # 能够保证取到正感兴趣区域.
    TRAIN_ROIS_PER_IMAGE = 4
    FPN_CLASSIF_FC_LAYERS_SIZE = 16
    # 每个epoch训练多少次
    STEPS_PER_EPOCH = 10

    # 每个epoch验证多少次
    VALIDATION_STEPS = 1

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
    # model_path = os.path.join(ROOT_DIR, "logs/cracks20220714T1112/mask_rcnn_cracks_0140.h5")
    # 用mini 最后一层
    model_path = os.path.join(ROOT_DIR, "logs/cracks20220814T1729/mask_rcnn_cracks_0001.h5")

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
    # 1为实例
    float_masks = r['float_masks'][:, :, :, 1]
    if len(float_masks) == 0:
        print("None detection")
        return None

    plt.figure("original_image",figsize=(12,8))
    plt.subplot(221),    plt.imshow(molded_image),    plt.title("original_image")
    plt.subplot(222),    plt.imshow(cvresize_img),    plt.title("cvresize_img")
    plt.subplot(223),    plt.imshow(original_mask),    plt.title("original_mask")
    plt.subplot(224),    plt.imshow(float_masks[0]),    plt.title("original_mask")
    plt.show()






    single_box_temp = np.reshape(r['rois'][0],[1,4])
    visualize.display_instances(original_image,r['rois'] , float_masks[0], r['class_ids'],
                                ["bg","crack"], r['scores'])
    mode_mask = utils.resize(original_mask, (DIM, DIM, 1))
    overlaps2 = utils.compute_overlaps_masks(mode_mask, float_masks[0])
    print(f'overlaps2 = {overlaps2}')

    return float_masks

    # END

if __name__ == '__main__':
    # print_hi('PyCharm')
    # load_infer_model(init_with_last = True)
    # load_infer_model()
    # check_dataset()
    # display_anchors()
    # train(140,init_with="last")
    # train(2,init_with="last")
    train(1)
    # det()
    # simple_det()
    # config.display()
    # det_crack500(min2 = True)
    # det_single('./test/test6')
    # det_single('./test/test5')
    # split_det("test5")
    # split_eval("test5")
    # split_detection()
