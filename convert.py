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
import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log
from main import CracksConfig,InferenceConfig

config = CracksConfig()
inference_config = InferenceConfig()

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

PATH_TO_SAVE_FROZEN_PB ="./"
FROZEN_NAME ="saved_model_bm_crack_test.pb"


def save_model(save_path, original_path="last"):
    """
    将训练的仅保存参数的h5文件转换为将整个model结构及参数保存的H5 model
    :param path: h5 model path
    :return:
    """

    test_config = InferenceConfig()
    model = modellib.MaskRCNN(config=test_config, mode="inference", model_dir=MODEL_DIR)
    if original_path == "last":
        original_path = model.find_last()

    model.load_weights(original_path, by_name=True)
    model.keras_model.save(save_path)


def load_model(Weights):
    global model, graph

    class InferenceConfig(Config):
        NAME = "crack"
        NUM_CLASSES = 1 + 1
        IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES
        #                 DETECTION_MAX_INSTANCES = 100
        #                 DETECTION_MIN_CONFIDENCE = 0.7
        #                 DETECTION_NMS_THRESHOLD = 0.3
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    Logs = "./logs"
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=Logs)
    Weights = Weights
    # model.find_last()
    model.load_weights(Weights, by_name=True)
    graph = tf.get_default_graph()


# Reference https://github.com/bendangnuksung/mrcnn_serving_ready/blob/master/main.py
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(model, name, sess):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.train.write_graph(frozen_graph, directory, name, as_text=False)


def keras_to_tflite(in_weight_file, out_weight_file, first = True):
    sess = tf.compat.v1.Session()
    keras.set_session(sess)
    # 实现在tf1.x环境下保存为pb文件
    # 再在tf2.x环境下进行转换
    # 因为第二轮要在tf2环境下不能加载模型
    first = first
    #         first = True

    if first:
        load_model(in_weight_file)
        global model
        tf.enable_control_flow_v2()
        freeze_model(model.keras_model, FROZEN_NAME, sess)
    # https://github.com/matterport/Mask_RCNN/issues/2020#issuecomment-596449757
    else:
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

