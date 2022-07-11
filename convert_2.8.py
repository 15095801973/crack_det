#2.8版本
import tensorflow as tf
import numpy as np
import os
import sys

# import mrcnn.model as modellib # https://github.com/matterport/Mask_RCNN/
import keras.backend as keras
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

PATH_TO_SAVE_FROZEN_PB ="./"
FROZEN_NAME ="saved_model_bm_crack_test.pb"


def keras_to_tflite(in_weight_file, out_weight_file, first = True):
    sess = tf.compat.v1.Session()
    keras.set_session(sess)
    # 实现在tf1.x环境下保存为pb文件
    # 再在tf2.x环境下进行转换
    # 因为第二轮要在tf2环境下不能加载模型
    first = first
    #         first = True
    input_arrays = ["input_image", "input_image_meta", "input_anchors"]
    output_arrays = ["mrcnn_detection/Reshape_1", "mrcnn_mask/Reshape_1"]
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        PATH_TO_SAVE_FROZEN_PB + "/" + FROZEN_NAME,
        input_arrays, output_arrays,
        input_shapes={"input_image": [1, 128, 128, 3], 'input_image_meta': [1, 14], 'input_anchors': [1, 4092, 4]}
    )
    #     converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
    #         PATH_TO_SAVE_FROZEN_PB+"/"+FROZEN_NAME,
    #         input_arrays, output_arrays,
    #         input_shapes={"input_image":[1,256,256,3]}
    #         )
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.post_training_quantize = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open(out_weight_file, "wb").write(tflite_model)
    print("*" * 80)
    print("Finished converting keras model to Frozen tflite")
    print('PATH: ', out_weight_file)
    print("*" * 80)

if __name__ == '__main__':
    print(tf.__version__)
    # save_model('mask_rcnn_crack_full.h5', original_path="last")
    # keras_to_tflite("./mask_rcnn_crack_full.h5","./mask_rcnn_crack_test.tflite")
    keras_to_tflite("./mask_rcnn_crack_full.h5","./mask_rcnn_crack_test.tflite",first=False)

