import cv2
import matplotlib.pyplot as plt
import numpy as np

# 定义matshow方法
from mrcnn.config import Config


def matshow(title='image', image=None, gray=False):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pass
        elif gray:
            # 转换成GRAY格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # 图片默认BGR通道，将突破转换成RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 使用这种方式显示图片可能会导致图片显示畸形
            # cv2.imshow('image', image)
    plt.figure()

    # 载入图像
    plt.imshow(image, cmap="gray")

    # 设置标题
    plt.title(title)

    plt.show()


if __name__ == '__main__':
    name = 'test4'
    # 读取灰度图
    im = cv2.imread(f'test/{name}.jpg', 0)
    mask = cv2.imread(f'test/{name}.png', 0)
    height, width = im.shape[:2]
    # 高斯滤波后再采用Otsu阈值
    core_size = np.min([height, width]) // Config.GaussianBlurFactor * 2 + 1

    # 读取彩色图
    # im = cv2.imread(f'test/{name}.jpg', 1)
    matshow('im', im)
    matshow('mask', mask)

    # 绘制直方图
    plt.hist(im.ravel(), 256, [0, 256])
    plt.show()

    # 均衡化处理
    im_equ1 = cv2.equalizeHist(im)
    matshow('im_equ1', im_equ1)



    blur = cv2.GaussianBlur(im, (core_size, core_size), 0)
    matshow('blur1', blur)

    ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matshow('th1', th1)

    blur2 = cv2.GaussianBlur(im_equ1, (core_size, core_size), 0)
    matshow('blur2', blur2)

    ret1, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matshow('th2', th2)

    ret1, th3 = cv2.threshold(im_equ1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matshow('th3', th3)
    # cv2.imwrite(f'./test/Ehist_{name}.jpg', im_equ1, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    # 绘制均衡化处理的直方图
    plt.hist(im_equ1.ravel(), 256, [0, 256])
    plt.show()
